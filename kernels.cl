#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define NSPEEDS         9
#define C_SQ (1.f / 3.f) /* square of speed of sound */
#define W0 (4.f / 9.f)   /* weighting factor */
#define W1 (1.f / 9.f)   /* weighting factor */
#define W2 (1.f / 36.f)  /* weighting factor */


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
  if (!obstacles[ii + jj* nx]
      && (cells[(3 * nx * ny) + ii + jj* nx] - w1) > 0.f
      && (cells[(6 * nx * ny) + ii + jj* nx] - w2) > 0.f
      && (cells[(7 * nx * ny) + ii + jj* nx] - w2) > 0.f)
  {
    /* increase 'east-side' densities */
    cells[(1 * nx *ny) + ii + jj* nx] += w1;
    cells[(5 * nx *ny) + ii + jj* nx] += w2;
    cells[(8 * nx *ny) + ii + jj* nx] += w2;
    /* decrease 'west-side' densities */
    cells[(3 * nx * ny) + ii + jj* nx] -= w1;
    cells[(6 * nx * ny) + ii + jj* nx] -= w2;
    cells[(7 * nx * ny) + ii + jj* nx] -= w2;
  }
}

kernel void propagate(global float* cells,
                      global float* tmp_cells,
                      global int* obstacles,
                      int nx, int ny)
{
  /* get column and row indices */
  int ii = get_global_id(0);
  int jj = get_global_id(1);

  /* determine indices of axis-direction neighbours
  ** respecting periodic boundary conditions (wrap around) */
  int y_n = (jj + 1) % ny;
  int x_e = (ii + 1) % nx;
  int y_s = (jj == 0) ? (jj + ny - 1) : (jj - 1);
  int x_w = (ii == 0) ? (ii + nx - 1) : (ii - 1);

  /* propagate densities from neighbouring cells, following
  ** appropriate directions of travel and writing into
  ** scratch space grid */
  tmp_cells[(0 * nx * ny) + ii + jj*nx] = cells[(0 * nx + ny) + ii + jj*nx]; /* central cell, no movement */
  tmp_cells[(1 * nx * ny) + ii + jj*nx] = cells[(1 * nx + ny) + x_w + jj*nx]; /* east */
  tmp_cells[(2 * nx * ny) + ii + jj*nx] = cells[(2 * nx + ny) + ii + y_s*nx]; /* north */
  tmp_cells[(3 * nx * ny) + ii + jj*nx] = cells[(3 * nx + ny) + x_e + jj*nx]; /* west */
  tmp_cells[(4 * nx * ny) + ii + jj*nx] = cells[(4 * nx + ny) + ii + y_n*nx]; /* south */
  tmp_cells[(5 * nx * ny) + ii + jj*nx] = cells[(5 * nx + ny) + x_w + y_s*nx]; /* north-east */
  tmp_cells[(6 * nx * ny) + ii + jj*nx] = cells[(6 * nx + ny) + x_e + y_s*nx]; /* north-west */
  tmp_cells[(7 * nx * ny) + ii + jj*nx] = cells[(7 * nx + ny) + x_e + y_n*nx]; /* south-west */
  tmp_cells[(8 * nx * ny) + ii + jj*nx] = cells[(8 * nx + ny) + x_w + y_n*nx]; /* south-east */
}

kernel void rebound(global float* cells, global float* tmp_cells, global int* obstacles,int nx, int ny)
{
  /* loop over the cells in the grid */
  int ii = get_global_id(0);
  int jj = get_global_id(1);

  /* if the cell contains an obstacle */
  if (obstacles[jj*nx + ii])
  {
    /* called after propagate, so taking values from scratch space
    ** mirroring, and writing into main grid */
    cells[(1 * nx * ny) + ii + jj*nx] = tmp_cells[(3 * nx * ny) + ii + jj*nx];
    cells[(2 * nx * ny) + ii + jj*nx] = tmp_cells[(4 * nx * ny) + ii + jj*nx];
    cells[(3 * nx * ny) + ii + jj*nx] = tmp_cells[(1 * nx * ny) + ii + jj*nx];
    cells[(4 * nx * ny) + ii + jj*nx] = tmp_cells[(2 * nx * ny) + ii + jj*nx];
    cells[(5 * nx * ny) + ii + jj*nx] = tmp_cells[(7 * nx * ny) + ii + jj*nx];
    cells[(6 * nx * ny) + ii + jj*nx] = tmp_cells[(8 * nx * ny) + ii + jj*nx];
    cells[(7 * nx * ny) + ii + jj*nx] = tmp_cells[(5 * nx * ny) + ii + jj*nx];
    cells[(8 * nx * ny) + ii + jj*nx] = tmp_cells[(6 * nx * ny) + ii + jj*nx];
  }
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

      /* don't consider occupied cells */
      if (!obstacles[ii + jj*nx])
      {
        /* compute local density total */
        float local_density = 0.f;

        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += tmp_cells[(kk * nx * ny) + ii + jj*nx];
        }

        /* compute x velocity component */
        float u_x = (tmp_cells[(1 * nx * ny) + ii + jj*nx]
                   + tmp_cells[(5 * nx * ny) + ii + jj*nx]
                   + tmp_cells[(8 * nx * ny) + ii + jj*nx]
                  - (tmp_cells[(3 * nx * ny) + ii + jj*nx]
                   + tmp_cells[(6 * nx * ny) + ii + jj*nx]
                   + tmp_cells[(7 * nx * ny) + ii + jj*nx]))
                     / local_density;
        /* compute y velocity component */
        float u_y = (tmp_cells[(2 * nx * ny) + ii + jj*nx]
                   + tmp_cells[(5 * nx * ny) + ii + jj*nx]
                   + tmp_cells[(6 * nx * ny) + ii + jj*nx]
                  - (tmp_cells[(4 * nx * ny) + ii + jj*nx]
                   + tmp_cells[(7 * nx * ny) + ii + jj*nx]
                   + tmp_cells[(8 * nx * ny) + ii + jj*nx]))
                     / local_density;

        /* velocity squared */
        float u_sq = u_x * u_x + u_y * u_y;

        /* directional velocity components */
        float u[NSPEEDS];
        u[1] =   u_x;        /* east */
        u[2] =         u_y;  /* north */
        u[3] = - u_x;        /* west */
        u[4] =       - u_y;  /* south */
        u[5] =   u_x + u_y;  /* north-east */
        u[6] = - u_x + u_y;  /* north-west */
        u[7] = - u_x - u_y;  /* south-west */
        u[8] =   u_x - u_y;  /* south-east */

        /* equilibrium densities */
        float d_equ[NSPEEDS];
        /* zero velocity density: weight w0 */
        d_equ[0] = w0 * local_density * (1.f - u_sq / (2.f * c_sq));
        /* axis speeds: weight w1 */
        d_equ[1] = W1 * local_density * (1.f + 3 * u[1] + 4.5f * (u[1] * u[1]) - 1.5f * u_sq);
        d_equ[2] = W1 * local_density * (1.f + 3 * u[2] + 4.5f * (u[2] * u[2]) - 1.5f * u_sq);
        d_equ[3] = W1 * local_density * (1.f + 3 * u[3] + 4.5f * (u[3] * u[3]) - 1.5f * u_sq);
        d_equ[4] = W1 * local_density * (1.f + 3 * u[4] + 4.5f * (u[4] * u[4]) - 1.5f * u_sq);
        /* diagonal speeds: weight w2 */
        d_equ[5] = W2 * local_density * (1.f + 3 * u[5] + 4.5f * (u[5] * u[5]) - 1.5f * u_sq);
        d_equ[6] = W2 * local_density * (1.f + 3 * u[6] + 4.5f * (u[6] * u[6]) - 1.5f * u_sq);
        d_equ[7] = W2 * local_density * (1.f + 3 * u[7] + 4.5f * (u[7] * u[7]) - 1.5f * u_sq);
        d_equ[8] = W2 * local_density * (1.f + 3 * u[8] + 4.5f * (u[8] * u[8]) - 1.5f * u_sq);

        /* relaxation step */
        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          cells[(kk * nx * ny) + ii + jj*nx] = tmp_cells[(kk * nx * ny) + ii + jj*nx]
                                                  + omega
                                                  * (d_equ[kk] - tmp_cells[(kk * nx * ny) + ii + jj*nx]);
        }
      }

}


kernel void av_velocity(global float* cells, global float* tot_u, global int* obstacles, local float* loc_u, int nx, int ny) {
  /* loop over the cells in the grid */
  int ii = get_global_id(0);
  int jj = get_global_id(1);
  int local_ii = get_local_id(0);
  int local_jj = get_local_id(1);
  int local_nx = get_local_size(0);
  int local_ny = get_local_size(1);
  int group_ii = get_group_id(0);
  int group_jj = get_group_id(1);

  loc_u[local_ii + (local_nx * local_jj)] = 0.f;

  if (!obstacles[ii + jj*nx]) {
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

    float u_y = (cells[(2 * nx * ny ) + ii + jj*nx] 
               + cells[(5 * nx * ny ) + ii + jj*nx] 
               + cells[(6 * nx * ny ) + ii + jj*nx] 
              - (cells[(4 * nx * ny ) + ii + jj*nx] 
               + cells[(7 * nx * ny ) + ii + jj*nx] 
               + cells[(8 * nx * ny ) + ii + jj*nx])) / local_density;
    
    loc_u[local_ii + (local_nx * local_jj)] = sqrt((u_x * u_x) + (u_y * u_y));
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  float group_sum = 0.f;

  if (local_ii == 1 && local_jj == 1){
    for (int i = 0; i < local_nx * local_ny; i++) {
        group_sum += loc_u[i];             
    }
    tot_u[group_ii + ((nx / local_nx) * group_jj)] = group_sum;                                       
  }
}