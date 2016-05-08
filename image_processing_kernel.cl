#define domain_skip_factor 2

typedef struct{
    int domain_x;
    int domain_y;
    int range_x;
    int range_y;
    float scale;
    int offset;
    int symmetry;
} transformation;

void transform_block(uchar* block, uchar* destination, int sym){
    int fromX = 0, fromY = 0;
    int dx = 1, dy = 1;
    bool inOrder = (sym == 0 || sym == 2 || sym == 4 || sym == 5);

    if (!(sym == 0 || sym == 1 || sym == 5 || sym == 6)){
        fromX += range_block_size - 1;
        dx = -1;
    }

    if (!(sym == 0 || sym == 3 || sym == 4 || sym == 6)){
        fromY += range_block_size - 1;
        dy = -1;
    }

    int startX = fromX;
    int startY = fromY;
    for (int toY = 0; toY <range_block_size; toY++){
		for (int toX = 0; toX < range_block_size; toX++){

			//int pixel = block[fromY * range_block_size + fromX];

			//if (pixel < 0)
			//	pixel = 0;
			//if (pixel > 255)
            //  pixel = 255;
			destination[toY * range_block_size + toX] =
			    block[fromY * range_block_size + fromX];

			if (inOrder)
				fromX += dx;
			else
				fromY += dy;
		}
		if (inOrder)
		{
			fromX = startX;
			fromY += dy;
		}
		else
		{
			fromY = startY;
			fromX += dx;
		}
	}
}

int get_block_average(uchar* block){
    int sum = 0;
    for(int i=0; i < range_block_size*range_block_size; i++){
        sum+=(int)block[i];
    }
    return sum/(range_block_size*range_block_size);
}

float get_scale_factor(uchar* range, uchar* domain, int range_average,
                       int domain_average){
    int top = 0, bottom = 0;
    for (int i=0; i<range_block_size*range_block_size; i++){
        int domain_part = (int)domain[i] - domain_average;
        int range_part = (int)range[i] - range_average;

        top += domain_part*range_part;
        bottom += domain_part*domain_part;
    }

    if (bottom == 0){
        return 0.f;
    }

    return top*1.f/bottom;
}

float get_error(uchar* range, uchar* domain, int range_average,
                int domain_average, float scale, int offset){
    float error = 0.f;

    for(int i=0; i <range_block_size*range_block_size; i++){
        int range_part = (int)range[i] - range_average;
        int domain_part = (int)domain[i] - domain_average;
        float difference = domain_part*scale - range_part;

        error += difference*difference;
    }

    return error/(range_block_size*range_block_size);
}

__const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
                          CLK_ADDRESS_CLAMP_TO_EDGE |
                          CLK_FILTER_NEAREST;

__kernel void find_matches(__read_only image2d_t image,
                           __read_only image2d_t downsampled,
                           __global int2* range_blocks_coordinates,
                           __const int image_width, __const int image_height,
                           __const int global_width,
                           __global transformation* transformations){
    int index = get_global_id(0);
    //int x = index % global_width, y = index / global_width;
    //if (x*range_block_size >= image_width-range_block_size+1 ||
    //    y*range_block_size >= image_height-range_block_size+1)
    //    return;
    uchar range[range_block_size*range_block_size] ;
    uchar domain[range_block_size*range_block_size];
    uchar domain_transformation[range_block_size*range_block_size];
    int2 range_coords = range_blocks_coordinates[index];
    for (int i=0; i < range_block_size; i++){
        for(int j=0; j < range_block_size; j++){
            range[i*range_block_size+j] = (uchar)read_imageui(
                image, sampler, (int2)(range_coords.x+j, range_coords.y+i)).x;
        }
    }

    int range_average = get_block_average(range);

    float min_error = (float)1e10;

    transformation best_transformation;
    best_transformation.range_x = range_coords.x;
    best_transformation.range_y = range_coords.y;

    for(int i=0; i<image_height/2-range_block_size+1; i+=domain_skip_factor){
        for(int j=0; j<image_width/2-range_block_size+1; j+=domain_skip_factor){
            for(int dy=0; dy<range_block_size; dy++){
                for(int dx=0; dx<range_block_size; dx++){

                    domain[dy*range_block_size+dx] = (uchar)read_imageui(downsampled,
                        sampler, (int2)(j+dx, i+dy)).x;
                }
            }
            int domain_average = get_block_average(domain);
            for(int symmetry=0; symmetry<8; symmetry++){
                transform_block(domain, domain_transformation, symmetry);
                float scale = get_scale_factor(range, domain_transformation,
                    range_average, domain_average);
                int offset = range_average-scale*domain_average;
                float error = get_error(range, domain_transformation, range_average, domain_average,
                    scale, offset);

                if (error < min_error){
                    min_error = error;
                    best_transformation.domain_x = j*2;
                    best_transformation.domain_y = i*2;
                    best_transformation.scale = scale;
                    best_transformation.offset = offset;
                    best_transformation.symmetry = symmetry;
                }
            }
        }
    }

    transformations[index] = best_transformation;
}