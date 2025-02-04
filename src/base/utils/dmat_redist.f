! This Source Code Form is subject to the terms of the Mozilla Public
! License, v. 2.0. If a copy of the MPL was not distributed with this
! file, You can obtain one at http://mozilla.org/MPL/2.0/.

! Copyright 2013, Schmidt


! Wrapper for pdgemr2d
! INPUTS
  ! X = Input submatrix.
  ! IX/JX = 
  ! DESCX = Descriptor array for X.
  ! IY/JY = 
  ! DESCY = Descriptor array for Y.
  ! CMNCTXT = Common BLACS context for X and Y.
! OUTPUTS
  ! Y = 
!!!      SUBROUTINE REDIST(X, IX, JX, DESCX, Y, IY, JY, DESCY, CMNCTXT)
!!!      IMPLICIT NONE
!!!      ! IN/OUT
!!!      INTEGER             IX, JX, DESCX(9), IY, JY, DESCY(9), CMNCTXT
!!!      DOUBLE PRECISION    X( * ), Y( * )
!!!      ! Local
!!!      INTEGER             M, N, MXLDM, DESCA(9),
!!!     $                    LDMX(2), LDMY(2), BLACSX(4), BLACSY(4)
!!!      ! External
!!!      EXTERNAL            PDGEMR2D
!!!      
!!!      
!!!      ! Get local and proc grid info
!!!      CALL PDIMS(DESCX, LDMX, BLACSX)
!!!      CALL PDIMS(DESCX, LDMY, BLACSY)
!!!      
!!!      M = DESCX(3)
!!!      N = DESCX(4)
!!!      
!!!      ! Adjust LDA since PDGEMR2D crashes all the time when LDA=1
!!!      DESCA(3) = 1
!!!      DESCA(4) = 1
!!!      DESCA(9) = 1
!!!      
!!!      MXLDM = MAX(LDMX)
!!!      DESCA(2) = DESCX(2)
!!!      CALL IALLREDUCE(MXLDM, DESCA, 'MAX', 'All')
!!!      IF (DESCX(9).EQ.1 .AND. DESCX(3).GT.1) DESCX(9) = MXLDM
!!!      
!!!      MXLDM = MAX(LDMY)
!!!      DESCA(2) = DESCY(2)
!!!      CALL IALLREDUCE(MXLDM, DESCA, 'MAX', 'All')
!!!      IF (DESCY(9).EQ.1 .AND. DESCY(3).GT.1) DESCY(9) = MXLDM
!!!      
!!!      ! Redistribute
!!!      CALL PDGEMR2D(M, N, X, IX, JX, DESCX,
!!!     $              Y, IY, JY, DESCY, CMNCTXT)
!!!      
!!!      RETURN
!!!      END SUBROUTINE


! Construct local submatrix from global matrix
! INPUTS
  ! GBLX = Global, non-distributed matrix.  Owned by which processor(s) depends 
    ! on R/CSRC values
  ! DESCX = ScaLAPACK descriptor array for SUBX (not a typo).
  ! RSRC/CSRC = Row/Column process value corresponding to BLACS grid for the
    ! value in DESCX(2) (the ICTXT) on which the data is stored.  If RSRC = -1,
    ! then CSRC is ignored and total ownership is assumed, i.e., GBLX is owned 
    ! by all processors.
! OUTPUTS
  ! SUBX = Local submatrix.
      SUBROUTINE MKSUBMAT(GBLX, SUBX, DESCX)!, RSRC, CSRC)
      IMPLICIT NONE
      ! IN/OUT
      INTEGER             DESCX(9)!, RSRC, CSRC
      DOUBLE PRECISION    GBLX(DESCX(3), DESCX(4)), SUBX(DESCX(9), *)
      ! Local
      INTEGER             M, N, I, J, GI, GJ, RBL, CBL, !TI, TJ,
     $                    LDM(2), BLACS(5)
      ! External
      EXTERNAL            PDIMS, L2GPAIR
      
      
      ! Get local and proc grid info
      CALL PDIMS(DESCX, LDM, BLACS)
      
      M = LDM(1)
      N = LDM(2)
      
      RBL = DESCX(5)
      CBL = DESCX(6)
      
      IF (M.GT.0 .AND. N.GT.0) THEN
        ! FIXME
        DO J = 1, N
          DO I = 1, M
            CALL L2GPAIR(I, J, GI, GJ, DESCX, BLACS)
            SUBX(I, J) = GBLX(GI, GJ)
          END DO 
        END DO
!        DO J = 1, N, CBL
!          DO I = 1, M, RBL
!            CALL L2GPAIR(I, J, GI, GJ, DESCX, BLACS)
!            
!            RBL = MIN(RBL, M-I+1)
!            CBL = MIN(CBL, N-J+1)
!            
!            DO TJ = 0, CBL-1
!              DO TI = 0, RBL-1
!                SUBX(I+TI, J+TJ) = GBLX(GI+TI, GJ+TJ)
!              END DO
!            END DO
!          END DO 
!        END DO
      END IF
      
      RETURN
      END SUBROUTINE

! Construct covariance matrix from global location matrix
! INPUTS
  ! GBLX = Global, non-distributed matrix.  Owned by which processor(s) depends 
    ! on R/CSRC values
  ! DESCX = ScaLAPACK descriptor array for SUBX (not a typo).
  ! RSRC/CSRC = Row/Column process value corresponding to BLACS grid for the
    ! value in DESCX(2) (the ICTXT) on which the data is stored.  If RSRC = -1,
    ! then CSRC is ignored and total ownership is assumed, i.e., GBLX is owned 
    ! by all processors.
! OUTPUTS
  ! SUBX = Local covariance submatrix.
      SUBROUTINE COVSUBMAT(MODEL, PARAM, GBLX, SUBX, DESCX)
      IMPLICIT NONE
      ! IN/OUT
      INTEGER             DESCX(9), MODEL
      DOUBLE PRECISION    GBLX(DESCX(3), DESCX(4)), SUBX(DESCX(9), *)
      DOUBLE PRECISION    PARAM(3), L1(3), L2(3)
      ! Local
      INTEGER             M, N, I, J, GI, GJ, RBL, CBL, !TI, TJ,
     $                    LDM(2), BLACS(5)
      ! External
      EXTERNAL            PDIMS, L2GPAIR, COVFUNC
      
      
      ! Get local and proc grid info
      CALL PDIMS(DESCX, LDM, BLACS)
      
      M = LDM(1)
      N = LDM(2)
      
      RBL = DESCX(5)
      CBL = DESCX(6)
      
      IF (M.GT.0 .AND. N.GT.0) THEN
        ! FIXME
        DO J = 1, N
          DO I = 1, M
            CALL L2GPAIR(I, J, GI, GJ, DESCX, BLACS)
            L1 = GBLX(GI, :)
            L2 = GBLX(GJ, :)
            CALL COVFUNC(MODEL, PARAM, L1, L2, SUBX(I, J))
          END DO 
        END DO
      END IF
      
      RETURN
      END SUBROUTINE


! Construct cross-covariance matrix from global location matrix
! INPUTS
  ! GBLX = Global, non-distributed matrix.  Owned by which processor(s) depends 
    ! on R/CSRC values
  ! DESCX = ScaLAPACK descriptor array for SUBX (not a typo).
  ! RSRC/CSRC = Row/Column process value corresponding to BLACS grid for the
    ! value in DESCX(2) (the ICTXT) on which the data is stored.  If RSRC = -1,
    ! then CSRC is ignored and total ownership is assumed, i.e., GBLX is owned 
    ! by all processors.
! OUTPUTS
  ! SUBX = Local cross-covariance submatrix.
      SUBROUTINE CROSSCOVSUBMAT(MODEL, PARAM, GBLX, SUBX, DESCX)
      IMPLICIT NONE
      ! IN/OUT
      INTEGER             DESCX(9), MODEL
      DOUBLE PRECISION    GBLX(DESCX(3), DESCX(4)), SUBX(DESCX(9), *)
      !DOUBLE PRECISION    SUBX(2 * DESCX(9), *)
      DOUBLE PRECISION    PARAM(6), L1(7), L2(7) 
      !DOUBLE PRECISION    PARAM1(3), PARAM2(3), PARAM12(3)

      ! Local
      INTEGER             M, N, I, J, GI, GJ, RBL, CBL, INDX, INDY,
     $                    LDM(2), BLACS(5)
      ! External
      EXTERNAL            PDIMS, L2GPAIR, COVFUNC
      
      
      ! Get local and proc grid info
      CALL PDIMS(DESCX, LDM, BLACS)
      
      M = LDM(1)
      N = LDM(2)
      
      RBL = DESCX(5)
      CBL = DESCX(6)

      !PARAM1(1) = PARAM(1)
      !PARAM1(2) = PARAM(3)
      !PARAM1(3) = PARAM(4)

      !PARAM2(1) = PARAM(2)
      !PARAM2(2) = PARAM(3)
      !PARAM2(3) = PARAM(5)

      !PARAM12(1) = PARAM(6) * SQRT(PARAM(1) * PARAM(2)) 
      !PARAM12(2) = PARAM(3)
      !PARAM12(3) = 0.5 * (PARAM(4) + PARAM(5))
      
      IF (M.GT.0 .AND. N.GT.0) THEN
        ! FIXME
        !DO J = 1, (2 * N), 2
          !DO I = 1, (2 * M), 2
        DO J = 1, N
          DO I = 1, M
            CALL L2GPAIR(I, J, GI, GJ, DESCX, BLACS)
            L1 = GBLX(GI, :)
            L2 = GBLX(GJ, :)
            !INDX = (I - 1) * 2 + 1
            !INDY = (J - 1) * 2 + 1
            CALL COVFUNC(MODEL, PARAM, L1, L2, SUBX(I, J))
            !INDX = (I - 1) * 2 + 2
            !INDY = (J - 1) * 2 + 2
            !CALL COVFUNC(MODEL, PARAM2, L1, L2, SUBX(INDX, INDY))
            !INDX = (I - 1) * 2 + 1
            !INDY = (J - 1) * 2 + 2
            !CALL COVFUNC(MODEL, PARAM12, L1, L2, SUBX(INDX, INDY))
            !INDX = (I - 1) * 2 + 2
            !INDY = (J - 1) * 2 + 1
            !CALL COVFUNC(MODEL, PARAM12, L1, L2, SUBX(INDX, INDY))
            !CALL COVFUNC(MODEL, PARAM1, L1, L2, SUBX(I, J))
            !CALL COVFUNC(MODEL, PARAM2, L1, L2, SUBX(I + 1, J + 1))
            !CALL COVFUNC(MODEL, PARAM12, L1, L2, SUBX(I, J + 1))
            !CALL COVFUNC(MODEL, PARAM12, L1, L2, SUBX(I + 1, J))
          END DO 
        END DO
      END IF
      
      RETURN
      END SUBROUTINE

! Construct global matrix from local submatrix.
! INPUTS
  ! SUBX = Local submatrix.
  ! DESCX = ScaLAPACK descriptor array for SUBX.
  ! RDEST/CDEST = Row/Column process value corresponding to BLACS grid for the
    ! value in DESCX(2) (the ICTXT) on which the global matrix GBLX will be 
    ! stored.  If RDEST = -1, then CDEST is ignored and total ownership is 
    ! assumed, i.e., GBLX is given to all processors.
! OUTPUTS
  ! GBLX = Global, non-distributed matrix.
      SUBROUTINE MKGBLMAT(GBLX, SUBX, DESCX, RDEST, CDEST)
      IMPLICIT NONE
      ! IN/OUT
      INTEGER             DESCX(9), RDEST, CDEST
      DOUBLE PRECISION    GBLX(DESCX(3), DESCX(4)), SUBX(DESCX(9), *)
      ! Local
      INTEGER             M, N, I, J, GI, GJ, RBL, CBL, !TI, TJ,
     $                    LDM(2), BLACS(5)
      ! Parameter
      DOUBLE PRECISION    ZERO
      PARAMETER ( ZERO = 0.0D0 )
      ! External
      EXTERNAL            PDIMS, L2GPAIR, DGSUM2D, DALLREDUCE
      
      
      ! Get local and proc grid info
      CALL PDIMS(DESCX, LDM, BLACS)
      
      M = LDM(1)
      N = LDM(2)
      
      GBLX = ZERO
      
      RBL = DESCX(5)
      CBL = DESCX(6)
      
      IF (M.GT.0 .AND. N.GT.0) THEN
        ! FIXME
!        DO J = 1, N, CBL
!          DO I = 1, M, RBL
!            CALL L2GPAIR(I, J, GI, GJ, DESCX, BLACS)
!            
!            RBL = MIN(RBL, M-I+1)
!            CBL = MIN(CBL, N-J+1)
!            
!            DO TJ = 0, CBL-1
!              DO TI = 0, RBL-1
!                GBLX(GI+TI, GJ+TJ) = SUBX(I+TI, J+TJ)
!              END DO
!            END DO
!          
!          END DO 
!        END DO
        DO J = 1, N
          DO I = 1, M
            CALL L2GPAIR(I, J, GI, GJ, DESCX, BLACS)
            GBLX(GI, GJ) = SUBX(I, J)
          
          END DO 
        END DO
      END IF
      
      ! Have to move to a common grid for the reduction
      ! EDIT: Not sure to understand why it's needed, maybe could be made an option
      ! DESCX(2) = 0
      
      IF (RDEST.EQ.-1) THEN
        CALL DALLREDUCE(GBLX, DESCX, 'S', 'All')
      ELSE
        CALL DREDUCE(GBLX, DESCX, 'S', RDEST, CDEST, 'All')
      END IF
      
      RETURN
      END SUBROUTINE

