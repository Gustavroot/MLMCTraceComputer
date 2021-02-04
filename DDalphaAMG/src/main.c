/*
 * Copyright (C) 2016, Matthias Rottmann, Artur Strebel, Simon Heybrock, Simone Bacchio, Bjoern Leder, Issaku Kanamori.
 * 
 * This file is part of the DDalphaAMG solver library.
 * 
 * The DDalphaAMG solver library is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * The DDalphaAMG solver library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * 
 * You should have received a copy of the GNU General Public License
 * along with the DDalphaAMG solver library. If not, see http://www.gnu.org/licenses/.
 * 
 */
 
#include "main.h"

global_struct g;
#ifdef HAVE_HDF5
Hdf5_fileinfo h5info;
#endif
struct common_thread_data *commonthreaddata;
struct Thread *no_threading;

int main( int argc, char **argv ) {
    
#ifdef HAVE_HDF5
  h5info.filename=NULL;
  h5info.file_id=-1; 
  h5info.rootgroup_id=-1; 
  h5info.configgroup_id=-1;
  h5info.eigenmodegroup_id=-1;
  h5info.thiseigenmodegroup_id=-1;
  h5info.isOpen=0;
  h5info.mode=-1;
#endif
  level_struct l;
  config_double hopp = NULL, clov = NULL;
  
  MPI_Init( &argc, &argv );
  
  predefine_rank();
  if ( g.my_rank == 0 ) {
    printf("\n\n+----------------------------------------------------------------------+\n");
    printf("| The DDalphaAMG solver library.                                       |\n");
    printf("| Copyright (C) 2016, Matthias Rottmann, Artur Strebel,                |\n");
    printf("|       Simon Heybrock, Simone Bacchio, Bjoern Leder, Issaku Kanamori. |\n");
    printf("|                                                                      |\n");
    printf("| This program comes with ABSOLUTELY NO WARRANTY.                      |\n");
    printf("+----------------------------------------------------------------------+\n\n");
  }
  
  method_init( &argc, &argv, &l );
  
  no_threading = (struct Thread *)malloc(sizeof(struct Thread));
  setup_no_threading(no_threading, &l);
  
  MALLOC( hopp, complex_double, 3*l.inner_vector_size );
  if ( g.two_cnfgs ) {
    MALLOC( clov, complex_double, 3*l.inner_vector_size );
    printf0("clover term configuration: %s", g.in_clov ); 

    if(g.in_format == _LIME)
      lime_read_conf( (double*)(clov), g.in_clov, &(g.plaq_clov) );
    else if(g.in_format == _MULTI)
      read_conf_multi( (double*)(clov), g.in, &(g.plaq_hopp), &l );
    else
      read_conf( (double*)(clov), g.in_clov, &(g.plaq_clov), &l );

    printf0("hopping term ");
  }

  if(g.in_format == _LIME)
    lime_read_conf( (double*)(hopp), g.in, &(g.plaq_hopp) );
  else if(g.in_format == _MULTI)
    read_conf_multi( (double*)(hopp), g.in, &(g.plaq_hopp), &l );
  else
    read_conf( (double*)(hopp), g.in, &(g.plaq_hopp), &l );

  if ( !g.two_cnfgs ) {
    g.plaq_clov = g.plaq_clov;
  }
  // store configuration, compute clover term
  dirac_setup( hopp, clov, &l );
  FREE( hopp, complex_double, 3*l.inner_vector_size );
  if ( g.two_cnfgs ) {
    FREE( clov, complex_double, 3*l.inner_vector_size );
  }

  commonthreaddata = (struct common_thread_data *)malloc(sizeof(struct common_thread_data));
  init_common_thread_data(commonthreaddata);
  
#pragma omp parallel num_threads(g.num_openmp_processes)
  {
    struct Thread threading;
    setup_threading(&threading, commonthreaddata, &l);
    setup_no_threading(no_threading, &l);
    
    // setup up initial MG hierarchy
    method_setup( NULL, &l, &threading );
    
    // iterative phase
    method_update( l.setup_iter, &l, &threading );

    int lvl_nr = atoi(argv[3]);
    //int col_nr = atoi(argv[4]);

    // d_plus_clover_float
    // apply_coarse_operator_float
    // interpolate_float

    // getting the right l
    level_struct* lb = &l;
    for ( int i=0; i<lvl_nr; i++ ) {
        lb = lb->next_level;
    }

    if (!strcmp(argv[2],"A")) {
      printf("PRINTER : Printing A for level : %d\n", lvl_nr);

      int v_size = lb->vector_size;

      if ( lvl_nr==0 ) {
        vector_double v_in  = calloc( v_size, sizeof(complex_double) );
        vector_double v_out = calloc( v_size, sizeof(complex_double) );

        for (int j=0; j<lb->inner_vector_size; j++) {

          memset( v_in, 0.0, v_size*sizeof(complex_double) );
          v_in[j] = 1.0;

          d_plus_clover_double( v_out, v_in, &(g.op_double), lb, no_threading );

          printf("PRINTER : val = ");
          for ( int i=0; i<lb->inner_vector_size; i++ ) {
              printf("%.16f+%.16fj -- ", creal(v_out[i]), cimag(v_out[i]));
          }
          printf("\n");
        }

        free(v_in);
        free(v_out);
      } else {
        vector_float v_in  = calloc( v_size, sizeof(complex_float) );
        vector_float v_out = calloc( v_size, sizeof(complex_float) );

        for (int j=0; j<lb->inner_vector_size; j++) {

          memset( v_in, 0.0, v_size*sizeof(complex_float) );
          v_in[j] = 1.0;

          apply_coarse_operator_float( v_out, v_in, lb->p_float.op, lb, no_threading );

          printf("PRINTER : val = ");
          for ( int i=0; i<lb->inner_vector_size; i++ ) {
              printf("%.16f+%.16fj -- ", creal(v_out[i]), cimag(v_out[i]));
          }
          printf("\n");
        }

        free(v_in);
        free(v_out);
      }

      //printf("PRINTER : %d\n", v_size);
      //printf("PRINTER : %d\n", lb->inner_vector_size);

      printf("\n");
    }

    if (!strcmp(argv[2],"P")) {
      printf("PRINTER : Printing P for level : %d\n", lvl_nr);

      int v_size2 = lb->vector_size;
      int v_size1 = lb->next_level->vector_size;

      vector_float v_in  = calloc( v_size1, sizeof(complex_float) );
      vector_float v_out = calloc( v_size2, sizeof(complex_float) );

      for (int j=0; j<lb->next_level->inner_vector_size; j++) {

        memset( v_in, 0.0, v_size1*sizeof(complex_float) );
        v_in[j] = 1.0;

        interpolate3_float( v_out, v_in, lb, no_threading );

        printf("PRINTER : val = ");
        for ( int i=0; i<lb->inner_vector_size; i++ ) {
            printf("%.16f+%.16fj -- ", creal(v_out[i]), cimag(v_out[i]));
        }
        printf("\n");
      }

      free(v_in);
      free(v_out);
    }
    
    solve_driver( &l, &threading );
  }
  
  finalize_common_thread_data(commonthreaddata);
  finalize_no_threading(no_threading);
  method_free( &l );
  method_finalize( &l );
  
  MPI_Finalize();
  
  return 0;
}
