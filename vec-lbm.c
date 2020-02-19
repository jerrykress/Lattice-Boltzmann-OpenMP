#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <assert.h>

#define NSPEEDS 9
#define FINALSTATEFILE "final_state.dat"
#define AVVELSFILE "av_vels.dat"

/* struct to hold the parameter values */
typedef struct
{
    int nx;           /* no. of cells in x-direction */
    int ny;           /* no. of cells in y-direction */
    int maxIters;     /* no. of iterations */
    int reynolds_dim; /* dimension for Reynolds number */
    float density;    /* density per link */
    float accel;      /* density redistribution */
    float omega;      /* relaxation parameter */
} t_param;

float *speeds[NSPEEDS];
float *t_speeds[NSPEEDS];

/* utility functions */
void die(const char *message, const int line, const char *file)
{
    fprintf(stderr, "Error at line %d of file %s:\n", line, file);
    fprintf(stderr, "%s\n", message);
    fflush(stderr);
    exit(EXIT_FAILURE);
}

void usage(const char *exe)
{
    fprintf(stderr, "Usage: %s <paramfile> <obstaclefile>\n", exe);
    exit(EXIT_FAILURE);
}

/* load params, allocate memory, load obstacles & initialise fluid particle densities */
int initialise(const char *paramfile, const char *obstaclefile, t_param *params, int **obstacles_ptr, float **av_vels_ptr)
{
    char message[1024]; /* message buffer */
    FILE *fp;           /* file pointer */
    int xx, yy;         /* generic array indices */
    int blocked;        /* indicates whether a cell is blocked by an obstacle */
    int retval;         /* to hold return value for checking */

    /* open & read parameter file */
    fp = fopen(paramfile, "r");
    if (fp == NULL)
        die(message, __LINE__, __FILE__);
    retval = fscanf(fp, "%d\n", &(params->nx));
    if (retval != 1)
        die("could not read param file: nx", __LINE__, __FILE__);
    retval = fscanf(fp, "%d\n", &(params->ny));
    if (retval != 1)
        die("could not read param file: ny", __LINE__, __FILE__);
    retval = fscanf(fp, "%d\n", &(params->maxIters));
    if (retval != 1)
        die("could not read param file: maxIters", __LINE__, __FILE__);
    retval = fscanf(fp, "%d\n", &(params->reynolds_dim));
    if (retval != 1)
        die("could not read param file: reynolds_dim", __LINE__, __FILE__);
    retval = fscanf(fp, "%f\n", &(params->density));
    if (retval != 1)
        die("could not read param file: density", __LINE__, __FILE__);
    retval = fscanf(fp, "%f\n", &(params->accel));
    if (retval != 1)
        die("could not read param file: accel", __LINE__, __FILE__);
    retval = fscanf(fp, "%f\n", &(params->omega));
    if (retval != 1)
        die("could not read param file: omega", __LINE__, __FILE__);
    fclose(fp);

    /* the map of obstacles */
    //*obstacles_ptr = calloc(params->nx*params->ny, sizeof(int)); //malloc(sizeof(int) * (params->ny * params->nx));
    *obstacles_ptr = (int *)_mm_malloc(sizeof(int) * params->nx * params->ny, 64);
    if (*obstacles_ptr == NULL)
        die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

    for (int jj = 0; jj < params->ny; jj++)
    {
        for (int ii = 0; ii < params->nx; ii++)
        {
            (*obstacles_ptr)[ii + jj * params->nx] = 0;
        }
    }

    /* initialise densities */
    float w0 = params->density * 4.f / 9.f;
    float w1 = params->density / 9.f;
    float w2 = params->density / 36.f;

    for (int dim = 0; dim < NSPEEDS; dim++)
    {
        speeds[dim] = (float *)_mm_malloc(sizeof(float) * params->nx * params->nx, 64);   //calloc(params->nx * params->ny, sizeof(float));
        t_speeds[dim] = (float *)_mm_malloc(sizeof(float) * params->nx * params->nx, 64); //calloc(params->nx * params->ny, sizeof(float));
    }

    for (int i = 0; i < params->nx * params->ny; i++)
    {
        speeds[0][i] = w0;
        speeds[1][i] = w1;
        speeds[2][i] = w1;
        speeds[3][i] = w1;
        speeds[4][i] = w1;
        speeds[5][i] = w2;
        speeds[6][i] = w2;
        speeds[7][i] = w2;
        speeds[8][i] = w2;
    }

    /* open the obstacle data file */
    fp = fopen(obstaclefile, "r");
    if (fp == NULL)
    {
        sprintf(message, "could not open input obstacles file: %s", obstaclefile);
        die(message, __LINE__, __FILE__);
    }

    /* read-in the blocked cells list */
    while ((retval = fscanf(fp, "%d %d %d\n", &xx, &yy, &blocked)) != EOF)
    {
        /* some checks */
        if (retval != 3)
            die("expected 3 values per line in obstacle file", __LINE__, __FILE__);
        if (xx < 0 || xx > params->nx - 1)
            die("obstacle x-coord out of range", __LINE__, __FILE__);
        if (yy < 0 || yy > params->ny - 1)
            die("obstacle y-coord out of range", __LINE__, __FILE__);
        if (blocked != 1)
            die("obstacle blocked value should be 1", __LINE__, __FILE__);
        /* assign to array */
        (*obstacles_ptr)[xx + yy * params->nx] = blocked;
    }

    /* and close the file */
    fclose(fp);

    *av_vels_ptr = (float *)_mm_malloc(sizeof(float) * params->maxIters, 64);

    return EXIT_SUCCESS;
}

int accelerate_flow(const t_param params, int *obstacles)
{
    /* compute weighting factors */
    float w1 = params.density * params.accel / 9.f;
    float w2 = params.density * params.accel / 36.f;

    /* modify the 2nd row of the grid */
    int jj = params.ny - 2;

    for (int ii = 0; ii < params.nx; ii++)
    {
        /* if the cell is not occupied and
    ** we don't send a negative density */
        if (!obstacles[ii + jj * params.nx] && (speeds[3][ii + jj * params.nx] - w1) > 0.f && (speeds[6][ii + jj * params.nx] - w2) > 0.f && (speeds[7][ii + jj * params.nx] - w2) > 0.f)
        {
            /* increase 'east-side' densities */
            speeds[1][ii + jj * params.nx] += w1;
            speeds[5][ii + jj * params.nx] += w2;
            speeds[8][ii + jj * params.nx] += w2;
            /* decrease 'west-side' densities */
            speeds[3][ii + jj * params.nx] -= w1;
            speeds[6][ii + jj * params.nx] -= w2;
            speeds[7][ii + jj * params.nx] -= w2;
        }
    }

    return EXIT_SUCCESS;
}

int propagate(const t_param params)
{
    /* loop over _all_ cells */
    for (int jj = 0; jj < params.ny; jj++)
    {
        for (int ii = 0; ii < params.nx; ii++)
        {
            /* determine indices of axis-direction neighbours
      ** respecting periodic boundary conditions (wrap around) */
            int y_n = (jj + 1) % params.ny;
            int x_e = (ii + 1) % params.nx;
            int y_s = (jj == 0) ? (jj + params.ny - 1) : (jj - 1);
            int x_w = (ii == 0) ? (ii + params.nx - 1) : (ii - 1);
            /* propagate densities from neighbouring cells, following
      ** appropriate directions of travel and writing into
      ** scratch space grid */
            t_speeds[0][ii + jj * params.nx] = speeds[0][ii + jj * params.nx];   /* central cell, no movement */
            t_speeds[1][ii + jj * params.nx] = speeds[1][x_w + jj * params.nx];  /* east */
            t_speeds[2][ii + jj * params.nx] = speeds[2][ii + y_s * params.nx];  /* north */
            t_speeds[3][ii + jj * params.nx] = speeds[3][x_e + jj * params.nx];  /* west */
            t_speeds[4][ii + jj * params.nx] = speeds[4][ii + y_n * params.nx];  /* south */
            t_speeds[5][ii + jj * params.nx] = speeds[5][x_w + y_s * params.nx]; /* north-east */
            t_speeds[6][ii + jj * params.nx] = speeds[6][x_e + y_s * params.nx]; /* north-west */
            t_speeds[7][ii + jj * params.nx] = speeds[7][x_e + y_n * params.nx]; /* south-west */
            t_speeds[8][ii + jj * params.nx] = speeds[8][x_w + y_n * params.nx]; /* south-east */
        }
    }

    return EXIT_SUCCESS;
}

int boom(const t_param params, int *obstacles)
{
    /* loop over the cells in the grid
  ** NB the collision step is called after
  ** the propagate step and so values of interest
  ** are in the scratch-space grid */
    for (int jj = 0; jj < params.ny; jj++)
    {
        for (int ii = 0; ii < params.nx; ii++)
        {
            int i = ii + jj * params.nx;
            /* don't consider occupied cells */
            if (obstacles[jj * params.nx + ii])
            {
                /* called after propagate, so taking values from scratch space
        ** mirroring, and writing into main grid */
                speeds[1][ii + jj * params.nx] = t_speeds[3][ii + jj * params.nx];
                speeds[2][ii + jj * params.nx] = t_speeds[4][ii + jj * params.nx];
                speeds[3][ii + jj * params.nx] = t_speeds[1][ii + jj * params.nx];
                speeds[4][ii + jj * params.nx] = t_speeds[2][ii + jj * params.nx];
                speeds[5][ii + jj * params.nx] = t_speeds[7][ii + jj * params.nx];
                speeds[6][ii + jj * params.nx] = t_speeds[8][ii + jj * params.nx];
                speeds[7][ii + jj * params.nx] = t_speeds[5][ii + jj * params.nx];
                speeds[8][ii + jj * params.nx] = t_speeds[6][ii + jj * params.nx];
            }

            if (!obstacles[i])
            {
                /* compute local density total */
                float local_density = 0.f;

                for (int kk = 0; kk < NSPEEDS; kk++)
                    local_density += t_speeds[kk][i];

                /* compute x velocity component */
                float u_x = (t_speeds[1][i] + t_speeds[5][i] + t_speeds[8][i] - (t_speeds[3][i] + t_speeds[6][i] + t_speeds[7][i])) / local_density;
                /* compute y velocity component */
                float u_y = (t_speeds[2][i] + t_speeds[5][i] + t_speeds[6][i] - (t_speeds[4][i] + t_speeds[7][i] + t_speeds[8][i])) / local_density;

                /* directional velocity components */
                float u[NSPEEDS];
                u[1] = u_x;        /* east */
                u[2] = u_y;        /* north */
                u[3] = -u_x;       /* west */
                u[4] = -u_y;       /* south */
                u[5] = u_x + u_y;  /* north-east */
                u[6] = -u_x + u_y; /* north-west */
                u[7] = -u_x - u_y; /* south-west */
                u[8] = u_x - u_y;  /* south-east */

                /* equilibrium densities */
                float d_equ[NSPEEDS];
                /* velocity squared */
                float u_sq = u_x * u_x + u_y * u_y;

                /* speeds + relaxation step */
                for (int kk = 0; kk < NSPEEDS; kk++)
                {
                    if (kk == 0)
                    {
                        d_equ[0] = (4.f / 9.f) * local_density * (1.f - 1.5f * u_sq);
                    }
                    else if (kk <= 4)
                    {
                        d_equ[kk] = (1.f / 9.f) * local_density * (1.f + 3.f * u[kk] + (4.5f * u[kk] * u[kk]) - 1.5f * u_sq);
                    }
                    else
                    {
                        d_equ[kk] = (1.f / 36.f) * local_density * (1.f + 3.f * u[kk] + (4.5f * u[kk] * u[kk]) - 1.5f * u_sq);
                    }
                    speeds[kk][i] = t_speeds[kk][i] + params.omega * (d_equ[kk] - t_speeds[kk][i]);
                }
            }
        }
    }

    return EXIT_SUCCESS;
}

int write_values(const t_param params, int *obstacles, float *av_vels);

/* finalise, including freeing up allocated memory */
int finalise(const t_param *params, int **obstacles_ptr, float **av_vels_ptr);

/* Sum all the densities in the grid.
** The total should remain constant from one timestep to the next. */
float total_density(const t_param params);

/* calculate Reynolds number */
float calc_reynolds(const t_param params, int *obstacles);

float av_velocity(const t_param params, int *obstacles)
{
    int tot_cells = 0; /* no. of cells used in calculation */
    float tot_u;       /* accumulated magnitudes of velocity for each cell */

    /* initialise */
    tot_u = 0.f;

    /* loop over all non-blocked cells */
    for (int jj = 0; jj < params.ny; jj++)
    {
        for (int ii = 0; ii < params.nx; ii++)
        {
            /* ignore occupied cells */
            int i = ii + jj * params.nx;
            if (!obstacles[ii + jj * params.nx])
            {
                /* local density total */
                float local_density = 0.f;

                for (int kk = 0; kk < NSPEEDS; kk++)
                {
                    local_density += speeds[kk][i];
                }

                /* x, y components of velocity */
                float u_x = (speeds[1][i] + speeds[5][i] + speeds[8][i] - (speeds[3][i] + speeds[6][i] + speeds[7][i])) / local_density;
                float u_y = (speeds[2][i] + speeds[5][i] + speeds[6][i] - (speeds[4][i] + speeds[7][i] + speeds[8][i])) / local_density;

                //printf("U_X: %2.5f %2.5f %2.5f %2.5f %2.5f %2.5f\n", speeds[1][i], speeds[5][i], speeds[8][i], speeds[3][i], speeds[6][i], speeds[7][i]);
                //if (ii > 12) exit(0);
                /* accumulate the norm of x- and y- velocity components */
                tot_u += sqrtf((u_x * u_x) + (u_y * u_y));
                /* increase counter of inspected cells */
                ++tot_cells;
            }
        }
    }

    //printf("Total_U: %5.7f", tot_u);
    return tot_u / (float)tot_cells;
}

/*
** main program: initialise, timestep loop, finalise
*/
int main(int argc, char *argv[])
{
    char *paramfile = NULL;
    char *obstaclefile = NULL; /* name of a the input obstacle file */
    t_param params;            /* struct to hold parameter values */
    int *obstacles = NULL;     /* grid indicating which cells are blocked */
    float *av_vels = NULL;     /* a record of the av. velocity computed for each timestep */
    struct timeval timstr;
    struct rusage ru;
    double tic, toc;
    double usrtim;
    double systim;

    /* parse the command line */
    if (argc != 3)
        usage(argv[0]);
    else
    {
        paramfile = argv[1];
        obstaclefile = argv[2];
    }

    /* initialise our data structures and load values from file */
    initialise(paramfile, obstaclefile, &params, &obstacles, &av_vels);

    //start timer.
    gettimeofday(&timstr, NULL);
    tic = timstr.tv_sec + (timstr.tv_usec / 1000000.0);

    for (int tt = 0; tt < params.maxIters; tt++)
    {
        accelerate_flow(params, obstacles);
        propagate(params);
        boom(params, obstacles);
        av_vels[tt] = av_velocity(params, obstacles);

#ifdef DEBUG
        printf("==timestep: %d==\n", tt);
        printf("av velocity: %.12E\n", av_vels[tt]);
        printf("tot density: %.12E\n", total_density(params));
#endif
    }

    //end timer.
    gettimeofday(&timstr, NULL);
    toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
    getrusage(RUSAGE_SELF, &ru);
    timstr = ru.ru_utime;
    usrtim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
    timstr = ru.ru_stime;
    systim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
    /* write final values and free memory */
    printf("==done==\n");
    printf("Reynolds number:\t\t%.12E\n", calc_reynolds(params, obstacles));
    printf("Elapsed time:\t\t\t%.6lf (s)\n", toc - tic);
    printf("Elapsed user CPU time:\t\t%.6lf (s)\n", usrtim);
    printf("Elapsed system CPU time:\t%.6lf (s)\n", systim);
    write_values(params, obstacles, av_vels);
    finalise(&params, &obstacles, &av_vels);
}

int finalise(const t_param *params, int **obstacles_ptr, float **av_vels_ptr)
{
    _mm_free(*obstacles_ptr);
    *obstacles_ptr = NULL;

    _mm_free(*av_vels_ptr);
    *av_vels_ptr = NULL;

    return EXIT_SUCCESS;
}

float calc_reynolds(const t_param params, int *obstacles)
{
    const float viscosity = (2.f / params.omega - 1.f) / 6.f;
    return av_velocity(params, obstacles) * params.reynolds_dim / viscosity;
}

float total_density(const t_param params)
{
    float total = 0.f; /* accumulator */

    for (int jj = 0; jj < params.ny; jj++)
    {
        for (int ii = 0; ii < params.nx; ii++)
        {
            for (int kk = 0; kk < NSPEEDS; kk++)
            {
                total += speeds[kk][ii + jj * params.nx];
            }
        }
    }

    return total;
}

int write_values(const t_param params, int *obstacles, float *av_vels)
{
    FILE *fp;                     /* file pointer */
    const float c_sq = 1.f / 3.f; /* sq. of speed of sound */
    float local_density;          /* per grid cell sum of densities */
    float pressure;               /* fluid pressure in grid cell */
    float u_x;                    /* x-component of velocity in grid cell */
    float u_y;                    /* y-component of velocity in grid cell */
    float u;                      /* norm--root of summed squares--of u_x and u_y */

    fp = fopen(FINALSTATEFILE, "w");

    if (fp == NULL)
        die("could not open file output file", __LINE__, __FILE__);

    for (int jj = 0; jj < params.ny; jj++)
    {
        for (int ii = 0; ii < params.nx; ii++)
        {
            /* an occupied cell */
            int i = ii + jj * params.nx;

            if (obstacles[ii + jj * params.nx])
            {
                u_x = u_y = u = 0.f;
                pressure = params.density * c_sq;
            }
            /* no obstacle */
            else
            {
                local_density = 0.f;

                for (int kk = 0; kk < NSPEEDS; kk++)
                {
                    local_density += speeds[kk][i];
                }

                /* compute x velocity component */
                u_x = (speeds[1][i] + speeds[5][i] + speeds[8][i] - (speeds[3][i] + speeds[6][i] + speeds[7][i])) / local_density;

                /* compute y velocity component */
                u_y = (speeds[2][i] + speeds[5][i] + speeds[6][i] - (speeds[4][i] + speeds[7][i] + speeds[8][i])) / local_density;
                /* compute norm of velocity */
                u = sqrtf((u_x * u_x) + (u_y * u_y));
                /* compute pressure */
                pressure = local_density * c_sq;
            }

            /* write to file */
            fprintf(fp, "%d %d %.12E %.12E %.12E %.12E %d\n", ii, jj, u_x, u_y, u, pressure, obstacles[ii * params.nx + jj]);
        }
    }

    fclose(fp);

    fp = fopen(AVVELSFILE, "w");

    if (fp == NULL)
        die("could not open file output file", __LINE__, __FILE__);

    for (int ii = 0; ii < params.maxIters; ii++)
        fprintf(fp, "%d:\t%.12E\n", ii, av_vels[ii]);

    fclose(fp);

    return EXIT_SUCCESS;
}
