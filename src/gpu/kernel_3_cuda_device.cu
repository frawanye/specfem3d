/*
!=====================================================================
!
!                          S p e c f e m 3 D
!                          -----------------
!
!    Main historical authors: Dimitri Komatitsch and Jeroen Tromp
!                             CNRS, France
!                      and Princeton University, USA
!                (there are currently many more authors!)
!                          (c) October 2017
!
! This program is free software; you can redistribute it and/or modify
! it under the terms of the GNU General Public License as published by
! the Free Software Foundation; either version 3 of the License, or
! (at your option) any later version.
!
! This program is distributed in the hope that it will be useful,
! but WITHOUT ANY WARRANTY; without even the implied warranty of
! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
! GNU General Public License for more details.
!
! You should have received a copy of the GNU General Public License along
! with this program; if not, write to the Free Software Foundation, Inc.,
! 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
!
!=====================================================================
*/


__global__ void kernel_3_cuda_device(realw_p veloc,
                                     realw_p accel,
                                     int size,
                                     realw deltatover2,
                                     realw_const_p rmassx,
                                     realw_const_p rmassy,
                                     realw_const_p rmassz) {

  int id = threadIdx.x + (blockIdx.x + blockIdx.y*gridDim.x)*blockDim.x;

  // because of block and grid sizing problems, there is a small
  // amount of buffer at the end of the calculation
  if (id < size) {
    realw ax = __builtin_nontemporal_load(&accel[3*id  ]) * __builtin_nontemporal_load(&rmassx[id]);
    realw ay = __builtin_nontemporal_load(&accel[3*id+1]) * __builtin_nontemporal_load(&rmassy[id]);
    realw az = __builtin_nontemporal_load(&accel[3*id+2]) * __builtin_nontemporal_load(&rmassz[id]);

    __builtin_nontemporal_store(ax, &accel[3*id]);
    __builtin_nontemporal_store(ay, &accel[3*id+1]);
    __builtin_nontemporal_store(az, &accel[3*id+2]);

    const realw x = __builtin_nontemporal_load(&veloc[3*id]);
    const realw y = __builtin_nontemporal_load(&veloc[3*id+1]);
    const realw z = __builtin_nontemporal_load(&veloc[3*id+2]);
    __builtin_nontemporal_store(x + deltatover2 * ax, &veloc[3*id]);
    __builtin_nontemporal_store(y + deltatover2 * ay, &veloc[3*id+1]);
    __builtin_nontemporal_store(z + deltatover2 * az, &veloc[3*id+2]);
  }
}


