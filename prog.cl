__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE| //Natural coordinates
                                CLK_ADDRESS_CLAMP_TO_EDGE| //Clamp to zeros
                                CLK_FILTER_NEAREST;

__kernel void negative(__global uchar* image, int w, int h, int padding, __global uchar* imageOut)
{

    int x = get_global_id(0);
    int y = get_global_id(1);
    int idx = y * (w*3 + padding) + x*3 ;

    if((x < w) && (y < h)) {// check if x and y are valid image coordinates
        imageOut[idx] = 255 -image[idx];
        imageOut[idx+1] = 255 -image[idx+1];
        imageOut[idx+2] = 255 -image[idx+2];
    }
}

__kernel void binarization(__read_only image2d_t image, __write_only image2d_t imageOut,  int w, int h, int t)
{

    int x = get_global_id(0);
    int y = get_global_id(1);

    if( (x >= 0) && (x < w) && (y >= 0) && (y < h)) // check if x and y are valid image coordinates
    {
        uint4 pixel = read_imageui(image, sampler, (int2)(x,y));

        int p_intensity = (int)( round(((int)pixel.x + pixel.y + pixel.z) / 3.0) ); // average of the 3 channels (RGB)

        if(p_intensity > t)
            write_imageui(imageOut, (int2)(x,y), (uint4)(255, 255, 255 , 0));
        else
            write_imageui(imageOut, (int2)(x,y), (uint4)(0, 0, 0 , 0));

    }
}

__kernel void brightnessContrast(__read_only image2d_t image, __write_only image2d_t imageOut,  int w, int h, int brigh, double contrast)
{

    int x = get_global_id(0);
    int y = get_global_id(1);

    if( (x >= 0) && (x < w) && (y >= 0) && (y < h)) // check if x and y are valid image coordinates
    {
        uint4 pixel = read_imageui(image, sampler, (int2)(x,y));
       
        pixel.x = contrast * pixel.x + brigh;
        pixel.y = contrast * pixel.y + brigh;
        pixel.z = contrast * pixel.z + brigh;

        pixel.x =  pixel.x < 0 ? 0 :  pixel.x;
        pixel.y =  pixel.y < 0 ? 0 :  pixel.y;
        pixel.z =  pixel.z < 0 ? 0 :  pixel.z;

        pixel.x =  pixel.x > 255 ? 255 :  pixel.x;
        pixel.y =  pixel.y > 255 ? 255 :  pixel.y;
        pixel.z =  pixel.z > 255 ? 255 :  pixel.z;

        write_imageui(imageOut, (int2)(x,y), pixel);
    }
}

__kernel void hough_tf(__read_only image2d_t image, __global uint *votes_matrix, __global float *costheta_values, __global float *sentheta_values, int max_rho, int max_theta, int w, int h)
{

    int x = get_global_id(0);
    int y = get_global_id(1);

    if( (x >= 0) && (x < w) && (y >= 0) && (y < h) ) // check if x and y are valid image coordinates
    {
        // read whole image or accessing the image in each iteration
        uint4 pixel_value = read_imageui(image, sampler, (int2)(x,y));

        if (pixel_value.x == 255) // check if it's a black pixel
        {
            for (int i = 0; i < max_theta; i++)
            {
                int rho = (int)( round(x * costheta_values[i] + y * sentheta_values[i]));
                int index = rho*max_theta + i;

                if(index < max_theta*max_rho)
                    atomic_add(&votes_matrix[index], 1);
                else
                    atomic_add(&votes_matrix[max_theta*max_rho], 1);
            }
        }
    }
}

__kernel void select_max_matrix(__global uint *votes_matrix, int max_rho, int max_theta, __global int *max_values_rho, __global int *max_values_theta, __global int *max_votes)
{
    int prev_max = 0;
    int rho = 0;
    int theta = 0;

    for (int i = 1; i < max_rho; i++)
    {
       for (int j = 20; j < max_theta-90; j++)
       {
            int index = i*max_theta + j;
            max_votes[0] = votes_matrix[index];
            atomic_max(&max_votes[0], prev_max);
            if (max_votes[0] > prev_max)
            { 
                rho=i;
                theta=j;
                prev_max = max_votes[0];
            }
       }
    }

    max_values_rho[0] = rho;
    max_values_theta[0] = theta-90;
    rho = 0;
    theta = 0;
    prev_max = 0;
    
    for (int i = 1; i < max_rho; i++)
    {
        for (int j = 120; j <= 160; j++)
        {
            int index = i*max_theta + j;
            max_votes[1] = votes_matrix[index];
            atomic_max(&max_votes[1], prev_max);
            
            if (max_votes[1] > prev_max)
            {
                rho = i;
                theta = j;
                prev_max = max_votes[1];
            }
        }
    }

    max_values_rho[1] = rho;
    max_values_theta[1] = theta-90;

}