#include <complex.h>
#include <pfft.h>

static void init_parameters(
    int argc, char **argv,
    ptrdiff_t *n, int *np, int *loops, int *inplace,
    unsigned *opt_flag, unsigned *tune_flag, unsigned *destroy_flag);
static void measure_pfft(
    const ptrdiff_t *n, int *np, MPI_Comm comm,
    int loops, int inplace, unsigned pfft_opt_flags);

int main(int argc, char **argv)
{
  int np[2], inplace, loops;
  ptrdiff_t n[3];
  unsigned opt, tune, destroy_input;

  /* Set size of FFT and process mesh */
  n[0] = 128; n[1] = 128; n[2] = 128;
  np[0] = 0; np[1] = 0;
  inplace = 0;
  opt     = PFFT_MEASURE;
  tune    = PFFT_NO_TUNE;
  destroy_input = PFFT_PRESERVE_INPUT;
  loops   = 1;

  /* Initialize MPI and PFFT */
  MPI_Init(&argc, &argv);
  pfft_init();

  /* set parameters by command line */
  init_parameters(argc, argv, n, np, &loops, &inplace, &opt, &tune, &destroy_input);

  measure_pfft(n, np, MPI_COMM_WORLD, loops, inplace, opt | tune | destroy_input);

  MPI_Finalize();
  return 0;
}


static void measure_pfft(
    const ptrdiff_t *n, int *np, MPI_Comm comm,
    int loops, int inplace, unsigned pfft_opt_flags
    )
{
  ptrdiff_t alloc_local;
  ptrdiff_t local_ni[3], local_i_start[3];
  ptrdiff_t local_no[3], local_o_start[3];
  double err=0.0, timer[4], max_timer[4];
  double *in;
  pfft_complex *out;
  pfft_plan plan_forw=NULL, plan_back=NULL;
  MPI_Comm comm_cart_2d;
  int nump, rank;
  pfft_timer mytimer, mytimer2;
  double *myt;
  double tf[5][loops],tb[5][loops];
  double tt[loops];
  double t0, t1;

  /* Create two-dimensional process grid of size np[0] x np[1], if possible */
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &nump);
  MPI_Dims_create(nump, 2, np);
  pfft_printf(comm, "\n n0 = %d", np[0]);
  pfft_printf(comm, "\n n1 = %d", np[1]);
  if( pfft_create_procmesh_2d(comm, np[0], np[1], &comm_cart_2d) ){
    pfft_fprintf(comm, stderr, "Error: This test file only works with %d processes.\n", np[0]*np[1]);
    return;
  }

  /* Get parameters of data distribution */
  alloc_local = pfft_local_size_dft_r2c_3d(n, comm_cart_2d, PFFT_TRANSPOSED_OUT,
      local_ni, local_i_start, local_no, local_o_start);

  /* Allocate memory */
  in  = pfft_alloc_real(2 * alloc_local);
  out = pfft_alloc_complex(alloc_local);

  /* Plan parallel forward FFT */
  timer[0] = -MPI_Wtime();
  plan_forw = pfft_plan_dft_r2c_3d(
      n, in, out, comm_cart_2d, PFFT_FORWARD, PFFT_TRANSPOSED_OUT| pfft_opt_flags);
  timer[0] += MPI_Wtime();

  /* Plan parallel backward FFT */
  timer[1] = -MPI_Wtime();
  plan_back = pfft_plan_dft_c2r_3d(
      n, out, in, comm_cart_2d, PFFT_BACKWARD, PFFT_TRANSPOSED_IN| pfft_opt_flags);
  timer[1] += MPI_Wtime();

  /* Initialize input with random numbers */
  pfft_init_input_real(3, n, local_ni, local_i_start,
      in);

  pfft_reset_timer(plan_forw);
  pfft_reset_timer(plan_back);

  timer[2] = timer[3] = 0;
  for(int t=0; t<loops; t++){
    /* execute parallel forward FFT */
    MPI_Barrier(MPI_COMM_WORLD);
    tt[t] = MPI_Wtime();

    timer[2] -= MPI_Wtime();
    pfft_execute(plan_forw);
    timer[2] += MPI_Wtime();

    /* clear the old input */
    pfft_clear_input_real(3, n, local_ni, local_i_start,
        in);

    /* execute parallel backward FFT */
    //MPI_Barrier(MPI_COMM_WORLD);
    timer[3] -= MPI_Wtime();
    pfft_execute(plan_back);
    timer[3] += MPI_Wtime();

    tt[t] = MPI_Wtime() - tt[t];

    /* Print pfft timer */
    //pfft_print_average_timer_adv(plan_forw, comm_cart_2d);
    //pfft_print_average_timer_adv(plan_back, comm_cart_2d);
    
    // MM
    // Here I create timers corresponding to the ones in r2c_guru_pencil.c
    // Timers are provided by pfft. I think they correspond as shown below
    // Uncheck pfft_print_average_timer_adv above to see more details.
    // Should check timers more in detail.
    mytimer = pfft_get_timer(plan_forw);
    myt = pfft_convert_timer2vec(mytimer);
    tf[0][t] = myt[7];
    tf[1][t] = myt[12];
    tf[2][t] = myt[6];
    tf[3][t] = myt[11];
    tf[4][t] = myt[5];

    mytimer = pfft_get_timer(plan_back);
    myt = pfft_convert_timer2vec(mytimer);
    tb[0][t] = myt[8];
    tb[1][t] = myt[13];
    tb[2][t] = myt[9];
    tb[3][t] = myt[14];
    tb[4][t] = myt[10];

    // Resetting timers each step
    pfft_reset_timer(plan_forw);
    pfft_reset_timer(plan_back);

    /* Scale data */
    for(ptrdiff_t l=0; l < local_ni[0] * local_ni[1] * local_ni[2]; l++)
      in[l] /= (n[0]*n[1]*n[2]);
  }
  timer[2] /= loops;
  timer[3] /= loops;


  /* Print optimization flags */
  pfft_printf(comm_cart_2d, "\nFlags = ");
  if(pfft_opt_flags & PFFT_TUNE)
    pfft_printf(comm_cart_2d, "PFFT_TUNE");
  else
    pfft_printf(comm_cart_2d, "PFFT_NO_TUNE");

  pfft_printf(comm_cart_2d, " | ");

  if(pfft_opt_flags & PFFT_ESTIMATE)
    pfft_printf(comm_cart_2d, "PFFT_ESTIMATE");
  else if(pfft_opt_flags & PFFT_PATIENT)
    pfft_printf(comm_cart_2d, "PFFT_PATIENT");
  else if(pfft_opt_flags & PFFT_EXHAUSTIVE)
    pfft_printf(comm_cart_2d, "PFFT_EXHAUSTIVE");
  else
    pfft_printf(comm_cart_2d, "PFFT_MEASURE");

  pfft_printf(comm_cart_2d, " | ");

  if(pfft_opt_flags & PFFT_DESTROY_INPUT)
    pfft_printf(comm_cart_2d, "PFFT_DESTROY_INPUT");
  else
    pfft_printf(comm_cart_2d, "PFFT_PRESERVE_INPUT");

  pfft_printf(comm_cart_2d, "\n");


  /* Print error of back transformed data */
  err = pfft_check_output_real(3, n, local_ni, local_i_start, in, comm_cart_2d);
  pfft_printf(comm_cart_2d, "Run %d loops of ", loops);
  if(inplace)
    pfft_printf(comm_cart_2d, "in-place");
  else
    pfft_printf(comm_cart_2d, "out-of-place");
  pfft_printf(comm_cart_2d, " forward and backward trafo of size n=(%td, %td, %td):\n", n[0], n[1], n[2]);

  MPI_Reduce(&timer, &max_timer, 4, MPI_DOUBLE, MPI_MAX, 0, comm_cart_2d);
  pfft_printf(comm_cart_2d, "tune_forw = %6.2e; tune_back = %6.2e, exec_forw = %6.2e, exec_back = %6.2e, error = %6.2e\n", max_timer[0], max_timer[1], max_timer[2], max_timer[3], err);


  MPI_Allreduce(MPI_IN_PLACE,tt,loops,MPI_DOUBLE,MPI_MAX,comm);
  MPI_Allreduce(MPI_IN_PLACE,tf,5*loops,MPI_DOUBLE,MPI_MAX,comm);
  MPI_Allreduce(MPI_IN_PLACE,tb,5*loops,MPI_DOUBLE,MPI_MAX,comm);

  if (rank == 0)
  {
      t0 = 1e8;
      t1 = 0.;
      for (int i=0; i<loops; i++)
      {
          t0 = (tt[i] < t0) ? tt[i] : t0;
          t1 += tt[i];
          for (int j=0; j<5; j++)
          {
            tf[j][0] = (tf[j][i] < tf[j][0]) ? tf[j][i] : tf[j][0];
            tb[j][0] = (tb[j][i] < tb[j][0]) ? tb[j][i] : tb[j][0];
          }
      }
      printf("Fastest=%.6e \n", t0);
      printf("Average=%.6e \n", t1/((double) loops));
      printf("r2c=%.6e \n", tf[0][0]);
      printf("fc2c1=%.6e \n", tf[2][0]);
      printf("fc2c2=%.6e \n", tf[4][0]);
      printf("bc2c2=%.6e \n", tb[4][0]);
      printf("bc2c1=%.6e \n", tb[2][0]);
      printf("bc2r=%.6e \n", tb[0][0]);
      printf("Alltoall_f0=%.6e\n", tf[1][0]);
      printf("Alltoall_f1=%.6e\n", tf[3][0]);
      printf("Alltoall_b1=%.6e\n", tb[3][0]);
      printf("Alltoall_b0=%.6e\n", tb[1][0]);
  }


  /* free mem and finalize */
  pfft_destroy_plan(plan_forw);
  pfft_destroy_plan(plan_back);
  MPI_Comm_free(&comm_cart_2d);
  pfft_free(out);
  pfft_free(in);
}

static void init_parameters(
    int argc, char **argv,
    ptrdiff_t *n, int *np, int *loops, int *inplace,
    unsigned *opt_flag, unsigned *tune_flag, unsigned *destroy_input_flag
    )
{
  int opt=0, tune=0, destroy_input=0;

  pfft_get_args(argc, argv, "-pfft_n", 3, PFFT_PTRDIFF_T, n);
  pfft_get_args(argc, argv, "-pfft_np", 2, PFFT_INT, np);
  pfft_get_args(argc, argv, "-pfft_loops", 1, PFFT_INT, loops);
  pfft_get_args(argc, argv, "-pfft_ip", 1, PFFT_INT, inplace);
  pfft_get_args(argc, argv, "-pfft_opt", 1, PFFT_INT, &opt);
  pfft_get_args(argc, argv, "-pfft_tune", 1, PFFT_INT, &tune);
  pfft_get_args(argc, argv, "-pfft_di", 1, PFFT_INT, &destroy_input);


  switch(opt){
    case 1: *opt_flag = PFFT_MEASURE; break;
    case 2: *opt_flag = PFFT_PATIENT; break;
    case 3: *opt_flag = PFFT_EXHAUSTIVE; break;
  }

  if(destroy_input)
    *destroy_input_flag = PFFT_DESTROY_INPUT;

  if(tune)
    *tune_flag = PFFT_TUNE;
}

