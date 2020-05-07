#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define NSPEEDS         9
#define C_SQ (1.f / 3.f) /* square of speed of sound */
#define W0 (4.f / 9.f)   /* weighting factor */
#define W1 (1.f / 9.f)   /* weighting factor */
#define W2 (1.f / 36.f)  /* weighting factor */
#define C1_5 1.5f
#define C1_ 1.f
#define C4_5 4.5f


kernel void accelerate_flow(global float* cells,
                            global int* obstacles,
                            int nx, int ny,
                            float density, float accel)
{
  /* compute weighting factors */
  float w1 = density * accel / 9.0;
  float w2 = density * accel / 36.0;

  /* modify the 2nd row of the grid */
  int jj = ny - 2;

  /* get column index */
  int ii = get_global_id(0);

  /* if the cell is not occupied and
  ** we don't send a negative density */
  int condition = (!obstacles[ii + jj* nx] && (cells[(3 * nx * ny) + ii + jj* nx] - w1) > 0.f
                              && (cells[(6 * nx * ny) + ii + jj* nx] - w2) > 0.f
                              && (cells[(7 * nx * ny) + ii + jj* nx] - w2) > 0.f);
  
  /* increase 'east-side' densities */
  cells[(1 * nx *ny) + ii + jj* nx] += (w1 * (condition));
  cells[(5 * nx *ny) + ii + jj* nx] += (w2 * (condition));
  cells[(8 * nx *ny) + ii + jj* nx] += (w2 * (condition));
  /* decrease 'west-side' densities */
  cells[(3 * nx * ny) + ii + jj* nx] -= (w1 * (condition));
  cells[(6 * nx * ny) + ii + jj* nx] -= (w2 * (condition));
  cells[(7 * nx * ny) + ii + jj* nx] -= (w2 * (condition));
  
}

kernel void collision(global float* cells, global float* tmp_cells, global int* obstacles,int nx, int ny, float omega) {
  const float c_sq = 1.f / 3.f; /* square of speed of sound */
  const float w0 = 4.f / 9.f;  /* weighting factor */
  const float w1 = 1.f / 9.f;  /* weighting factor */
  const float w2 = 1.f / 36.f; /* weighting factor */

  /* loop over the cells in the grid
  ** NB the collision step is called after
  ** the propagate step and so values of interest
  ** are in the scratch-space grid */
  int ii = get_global_id(0);
  int jj = get_global_id(1);

  float tp[NSPEEDS];

  int y_n = (jj + 1) % ny;
  int x_e = (ii + 1) % nx;
  int y_s = (jj == 0) ? (jj + ny - 1) : (jj - 1);
  int x_w = (ii == 0) ? (ii + nx - 1) : (ii - 1);

  tp[0] = cells[(0 * nx * ny) + ii + jj*nx];
  tp[1] = cells[(1 * nx * ny) + x_w + jj*nx];
  tp[2] = cells[(2 * nx * ny) + ii + y_s*nx];
  tp[3] = cells[(3 * nx * ny) + x_e + jj*nx];
  tp[4] = cells[(4 * nx * ny) + ii + y_n*nx];
  tp[5] = cells[(5 * nx * ny) + x_w + y_s*nx];
  tp[6] = cells[(6 * nx * ny) + x_e + y_s*nx];
  tp[7] = cells[(7 * nx * ny) + x_e + y_n*nx];
  tp[8] = cells[(8 * nx * ny) + x_w + y_n*nx];

  /* compute local density total */
  float local_density = tp[0]
                      + tp[1]
                      + tp[2]
                      + tp[3]
                      + tp[4]
                      + tp[5]
                      + tp[6]
                      + tp[7]
                      + tp[8];

  /* compute x velocity component */
  float u_x = (tp[1] + tp[5] + tp[8] - (tp[3] + tp[6] + tp[7])) / local_density;
  /* compute y velocity component */
  float u_y = (tp[2] + tp[5] + tp[6] - (tp[4] + tp[7] + tp[8])) / local_density;

  /* velocity squared */
  float u_sq = u_x * u_x + u_y * u_y;

  /* directional velocity components */
  float u[NSPEEDS];
  u[0] = 0;
  u[1] = u_x;        /* east */
  u[2] = u_y;        /* north */
  u[3] = -u_x;       /* west */
  u[4] = -u_y;       /* south */
  u[5] = u_x + u_y;  /* north-east */
  u[6] = -u_x + u_y; /* north-west */
  u[7] = -u_x - u_y; /* south-west */
  u[8] = u_x - u_y;  /* south-east */

  tmp_cells[(0 * nx * ny) + ii + jj * nx] = (obstacles[jj * nx + ii]) ? tp[0] : tp[0] + omega * ((W0 * local_density * (C1_ - C1_5 * u_sq)) - tp[0]);
  tmp_cells[(1 * nx * ny) + ii + jj * nx] = (obstacles[jj * nx + ii]) ? tp[3] : tp[1] + omega * ((W1 * local_density * (C1_ + 3 * u[1] + C4_5 * (u[1] * u[1]) - C1_5 * u_sq)) - tp[1]);
  tmp_cells[(2 * nx * ny) + ii + jj * nx] = (obstacles[jj * nx + ii]) ? tp[4] : tp[2] + omega * ((W1 * local_density * (C1_ + 3 * u[2] + C4_5 * (u[2] * u[2]) - C1_5 * u_sq)) - tp[2]);
  tmp_cells[(3 * nx * ny) + ii + jj * nx] = (obstacles[jj * nx + ii]) ? tp[1] : tp[3] + omega * ((W1 * local_density * (C1_ + 3 * u[3] + C4_5 * (u[3] * u[3]) - C1_5 * u_sq)) - tp[3]);
  tmp_cells[(4 * nx * ny) + ii + jj * nx] = (obstacles[jj * nx + ii]) ? tp[2] : tp[4] + omega * ((W1 * local_density * (C1_ + 3 * u[4] + C4_5 * (u[4] * u[4]) - C1_5 * u_sq)) - tp[4]);
  tmp_cells[(5 * nx * ny) + ii + jj * nx] = (obstacles[jj * nx + ii]) ? tp[7] : tp[5] + omega * ((W2 * local_density * (C1_ + 3 * u[5] + C4_5 * (u[5] * u[5]) - C1_5 * u_sq)) - tp[5]);
  tmp_cells[(6 * nx * ny) + ii + jj * nx] = (obstacles[jj * nx + ii]) ? tp[8] : tp[6] + omega * ((W2 * local_density * (C1_ + 3 * u[6] + C4_5 * (u[6] * u[6]) - C1_5 * u_sq)) - tp[6]);
  tmp_cells[(7 * nx * ny) + ii + jj * nx] = (obstacles[jj * nx + ii]) ? tp[5] : tp[7] + omega * ((W2 * local_density * (C1_ + 3 * u[7] + C4_5 * (u[7] * u[7]) - C1_5 * u_sq)) - tp[7]);
  tmp_cells[(8 * nx * ny) + ii + jj * nx] = (obstacles[jj * nx + ii]) ? tp[6] : tp[8] + omega * ((W2 * local_density * (C1_ + 3 * u[8] + C4_5 * (u[8] * u[8]) - C1_5 * u_sq)) - tp[8]);

}


kernel void av_velocity(global float* cells, global float* tot_u, global int* obstacles, global float* av_vels, local float* loc_u, int nx, int ny, int tt, int tot_cells) {
  /* loop over the cells in the grid */
  int ii = get_global_id(0);
  int jj = get_global_id(1);
  int local_ii = get_local_id(0);
  int local_jj = get_local_id(1);
  int local_nx = get_local_size(0);
  int local_ny = get_local_size(1);
  int group_ii = get_group_id(0);
  int group_jj = get_group_id(1);

  int condition = (!obstacles[ii + jj*nx]);
  float local_density = 0.f;

  for (int kk = 0; kk < NSPEEDS; kk++) {
    local_density += cells[(kk * nx * ny) + ii + jj*nx];
  }

  float u_x = (cells[(1 * nx * ny) + ii + jj*nx]
              + cells[(5 * nx * ny) + ii + jj*nx]
              + cells[(8 * nx * ny) + ii + jj*nx] 
            - (cells[(3 * nx * ny) + ii + jj*nx] 
              + cells[(6 * nx * ny) + ii + jj*nx] 
              + cells[(7 * nx * ny) + ii + jj*nx])) / local_density;

  float u_y = (cells[(2 * nx * ny) + ii + jj*nx] 
              + cells[(5 * nx * ny) + ii + jj*nx] 
              + cells[(6 * nx * ny) + ii + jj*nx] 
            - (cells[(4 * nx * ny) + ii + jj*nx] 
              + cells[(7 * nx * ny) + ii + jj*nx] 
              + cells[(8 * nx * ny) + ii + jj*nx])) / local_density;
  
  loc_u[local_ii + (local_nx * local_jj)] = sqrt((u_x * u_x) + (u_y * u_y)) * condition;

  // Loop for computing localSums : divide WorkGroup into 2 parts
  for (uint stride = (local_nx * local_ny)/2; stride > 0; stride /=2)
  {
      // Waiting for each 2x2 addition into given workgroup
      barrier(CLK_LOCAL_MEM_FENCE);

      // Add elements 2 by 2 between local_id and local_id + stride
      if (local_ii + (local_nx * local_jj) < stride){
        loc_u[local_ii + (local_nx * local_jj)] +=  loc_u[local_ii + stride + (local_nx * (local_jj))];
      }
  }

  // Write result into partialSums[nWorkGroups]
  if (local_ii + (local_nx * local_jj) == 0){
    tot_u[group_ii + ((nx / local_nx) * group_jj)] = loc_u[0] / tot_cells; 
  }
}

kernel void sumGPU (global float* input, global float *partialSums, local float *localSums, global float* av_vels, int nx, int ny, int tt)
{
  int ii = get_global_id(0);
  int jj = get_global_id(1);
  int local_ii = get_local_id(0);
  int local_jj = get_local_id(1);
  int local_nx = get_local_size(0);
  int local_ny = get_local_size(1);
  nx = get_global_size(0);
  ny = get_global_size(1);
  int group_ii = get_group_id(0);
  int group_jj = get_group_id(1);

  localSums[local_ii + (local_nx * local_jj)] = input[ii + nx * jj];

// Loop for computing localSums : divide WorkGroup into 2 parts
  for (uint stride = (local_nx * local_ny)/2; stride > 0; stride /=2)
  {
      // Waiting for each 2x2 addition into given workgroup
      barrier(CLK_LOCAL_MEM_FENCE);

      // Add elements 2 by 2 between local_id and local_id + stride
      if (local_ii + (local_nx * local_jj) < stride){
        localSums[local_ii + (local_nx * local_jj)] +=  localSums[local_ii + stride + (local_nx * (local_jj))];
      }
  }

  // Write result into partialSums[nWorkGroups]
  if (local_ii + (local_nx * local_jj) == 0){
    partialSums[group_ii + ((nx / local_nx) * group_jj)] = localSums[0]; 
  }

  float sum = 0;

  for (int i = 0; i < (nx / local_nx) * (ny / local_ny); i++){
    sum += partialSums[i];
  }

  av_vels[tt] = sum;

}            