import cv2
import pyopencl as cl
import numpy as np
import math


def setup():
    try:
        plaforms = cl.get_platforms()
        global plaform
        plaform = plaforms[0]

        devices = plaform.get_devices()
        global device
        device = devices[0]

        global ctx
        ctx = cl.Context(devices)

        global commQ
        commQ = cl.CommandQueue(ctx, device)

        file = open("prog.cl", "r")
        global prog
        prog = cl.Program(ctx, file.read())
        prog.build()
        return True
    except Exception as e:
        print(e)
        return False


def region(image):
    polygon = np.array([
        [(200, 142), (50, 400), (710, 400), (540, 142)]

    ])

    mask = np.zeros_like(image)
    mask = cv2.fillPoly(mask, polygon, (255, 255, 255))
    mask = cv2.bitwise_and(image, mask)
    return mask


def lin_equ(l1, l2):
    m = (float)((l2[1] - l1[1]) / (l2[0] - l1[0]))
    c = (l2[1] - (m * l2[0]))
    return m, c


if setup():

    # Video Capture
    pathname = "Images/"
    car_cascade = cv2.CascadeClassifier(pathname + 'cars3.xml')

    vidCap = cv2.VideoCapture("video1.MTS")

    img_width = np.int32(720)
    img_height = np.int32(540)
    max_rho = np.int32(math.sqrt(math.pow(img_width, 2) + math.pow(img_height, 2)))
    max_theta = np.int32(180)
    votes_matrix = np.zeros((max_rho, max_theta), dtype=np.int32)

    costheta_values = np.cos(np.arange(-np.pi / 2, np.pi / 2, np.pi / 180), dtype=np.float32)
    sentheta_values = np.sin(np.arange(-np.pi / 2, np.pi / 2, np.pi / 180), dtype=np.float32)

    max_values_rho = np.zeros((2, 1), dtype=np.int32)
    max_values_theta = np.zeros((2, 1), dtype=np.int32)
    max_votes = np.zeros((2, 1), dtype=np.int32)
    threshold = np.int32(175)

    if not vidCap.isOpened():
        print("Video File Not Found")
        exit(-1)

    while True:

        ret, vidFrame = vidCap.read()
        if not ret:
            break
        vidFrame = cv2.resize(vidFrame, (720, 540))
        crop_img = region(vidFrame)

        imgIn_new = cv2.cvtColor(crop_img, cv2.COLOR_BGR2BGRA)
        imgOut = np.copy(imgIn_new)

        img_width = np.int32(imgIn_new.shape[1])
        img_height = np.int32(imgIn_new.shape[0])

        ############################# Binarization ######################################
        kernelName = prog.binarization

        imgFormat = cl.ImageFormat(cl.channel_order.BGRA, cl.channel_type.UNSIGNED_INT8)
        imgInBuffer = cl.Image(ctx, flags=cl.mem_flags.COPY_HOST_PTR | cl.mem_flags.READ_ONLY,
                               format=imgFormat,
                               shape=(img_width, img_height),
                               pitches=(imgIn_new.strides[0], imgIn_new.strides[1]),
                               hostbuf=imgIn_new.data)

        imgOutBuffer = cl.Image(ctx, flags=cl.mem_flags.ALLOC_HOST_PTR | cl.mem_flags.WRITE_ONLY,
                                format=imgFormat,
                                shape=(img_width, img_height))

        kernelName.set_arg(0, imgInBuffer)
        kernelName.set_arg(1, imgOutBuffer)  # Width
        kernelName.set_arg(2, img_width)  # Width
        kernelName.set_arg(3, img_height)  # Height
        kernelName.set_arg(4, threshold)  # threshold

        block_size = 32
        workGroupSize = (
            math.ceil(np.int32(imgIn_new.shape[1]) / block_size) * block_size,
            math.ceil(np.int32(imgIn_new.shape[0]) / block_size) * block_size)
        workItemSize = (block_size, block_size)  # 1024

        kernelEvent = cl.enqueue_nd_range_kernel(commQ, kernelName, global_work_size=workGroupSize,
                                                 local_work_size=workItemSize)
        kernelEvent.wait()

        cl.enqueue_copy(commQ, imgOut, imgOutBuffer, origin=(0, 0), region=(img_width, img_height))

        imgInBuffer.release()
        imgOutBuffer.release()

        ########################################### Hough Transform ###########################################################
        kernelName = prog.hough_tf

        imgInBuffer = cl.Image(ctx, flags=cl.mem_flags.COPY_HOST_PTR | cl.mem_flags.READ_ONLY,
                               format=imgFormat,
                               shape=(img_width, img_height),
                               pitches=(imgOut.strides[0], imgOut.strides[1]),
                               hostbuf=imgOut.data)

        costhetaBuffer = cl.Buffer(ctx, flags=cl.mem_flags.COPY_HOST_PTR | cl.mem_flags.READ_ONLY,
                                   hostbuf=costheta_values)

        senthetaBuffer = cl.Buffer(ctx, flags=cl.mem_flags.COPY_HOST_PTR | cl.mem_flags.READ_ONLY,
                                   hostbuf=sentheta_values)

        votesBuffer = cl.Buffer(ctx, flags=cl.mem_flags.COPY_HOST_PTR | cl.mem_flags.READ_WRITE,
                                hostbuf=votes_matrix)

        kernelName.set_arg(0, imgInBuffer)
        kernelName.set_arg(1, votesBuffer)
        kernelName.set_arg(2, costhetaBuffer)
        kernelName.set_arg(3, senthetaBuffer)
        kernelName.set_arg(4, max_rho)
        kernelName.set_arg(5, max_theta)
        kernelName.set_arg(6, img_width)  # Width
        kernelName.set_arg(7, img_height)  # Height

        workGroupSize = (
            math.ceil(np.int32(imgOut.shape[1]) / 32) * 32, math.ceil(np.int32(imgOut.shape[0]) / 32) * 32)
        workItemSize = (32, 32)  # 1024

        kernelEvent = cl.enqueue_nd_range_kernel(commQ, kernelName, global_work_size=workGroupSize,
                                                 local_work_size=workItemSize)
        kernelEvent.wait()

        ######################################## Select Max ######################################################################
        # Select max rho and theta
        kernelName = prog.select_max_matrix

        max_rho_buff = cl.Buffer(ctx, flags=cl.mem_flags.COPY_HOST_PTR | cl.mem_flags.READ_WRITE,
                                 hostbuf=max_values_rho)

        max_theta_buff = cl.Buffer(ctx, flags=cl.mem_flags.COPY_HOST_PTR | cl.mem_flags.READ_WRITE,
                                   hostbuf=max_values_theta)
        max_votes_buff = cl.Buffer(ctx, flags=cl.mem_flags.COPY_HOST_PTR | cl.mem_flags.READ_WRITE,
                                   hostbuf=max_votes)

        kernelName.set_arg(0, votesBuffer)
        kernelName.set_arg(1, max_rho)
        kernelName.set_arg(2, max_theta)
        kernelName.set_arg(3, max_rho_buff)
        kernelName.set_arg(4, max_theta_buff)
        kernelName.set_arg(5, max_votes_buff)

        workGroupSize = (1, 1)
        workItemSize = (1, 1)

        kernelEvent = cl.enqueue_nd_range_kernel(commQ, kernelName, global_work_size=workGroupSize,
                                                 local_work_size=workItemSize)
        kernelEvent.wait()

        cl.enqueue_copy(commQ, max_values_rho, max_rho_buff)
        cl.enqueue_copy(commQ, max_values_theta, max_theta_buff)
        a = math.cos(math.radians(max_values_theta[0]))
        b = math.sin(math.radians(max_values_theta[0]))
        x0 = a * max_values_rho[0]
        y0 = b * max_values_rho[0]
        pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
        pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
        m_blue, b_blue = lin_equ(pt1, pt2)
        if m_blue != 0:
            cv2.line(vidFrame, pt1, (int((70 - b_blue) / m_blue), 70), (255, 0, 0), 2, cv2.LINE_AA)
        else:
            cv2.line(vidFrame, pt1, pt2, (255, 0, 0), 2, cv2.LINE_AA)

        a = math.cos(math.radians(max_values_theta[1]))
        b = math.sin(math.radians(max_values_theta[1]))
        x0 = a * max_values_rho[1]
        y0 = b * max_values_rho[1]
        pt3 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
        pt4 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
        m_red, b_red = lin_equ(pt3, pt4)
        if m_red != 0:
            cv2.line(vidFrame, (int((420 - b_red) / m_red), 420), (int((70 - b_red) / m_red), 70), (0, 0, 255), 2,
                     cv2.LINE_AA)
        else:
            cv2.line(vidFrame, pt3, pt4, (255, 0, 0), 2, cv2.LINE_AA)

        crop_img_car = vidFrame[0:410, 50:50 + 500]  # ROI
        imgGray = cv2.cvtColor(crop_img_car, cv2.COLOR_BGR2GRAY)

        cars = car_cascade.detectMultiScale(imgGray, 1.4, 3)
        for (x, y, w, h) in cars:
            if (y + h) >= 90:
                cX = int((x + x + w) / 2.0)
                cY = int((y + y + h) / 2.0)
                if (cY) + m_red * (cX) + b_red > 0 and (cY) + m_blue * (cX) + b_blue > 0:
                    # print(x, y, w, h)
                    # print("red:", m_red, b_red)
                    # print("blue:", m_blue, b_blue)
                    cv2.rectangle(crop_img_car, (x, y), (x + w, y + h), (0, 0, 255), 2)
                else:
                    cv2.rectangle(crop_img_car, (x, y), (x + w, y + h), (0, 255, 0), 2)

        imgInBuffer.release()
        votesBuffer.release()
        max_rho_buff.release()
        max_theta_buff.release()

        cv2.imshow('Car and Lane detection', vidFrame)
        if cv2.waitKey(10) & 0xFF == ord('q'):  # 'q' key to close window
            break

    vidCap.release()
    cv2.destroyAllWindows()