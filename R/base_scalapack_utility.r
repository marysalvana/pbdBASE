# R version of ScaLAPACK tool DESCINIT
# Creates ScaLAPACK descriptor array for distributed matrix

#' @export
base.descinit <- function(dim, bldim, ldim, ICTXT=0)
{
  grid <- base.blacs(ICTXT)
  
  desc <- integer(9)
  
###  desc[1L] <- 1L                    # matrix type
###  desc[2L] <- ICTXT                 # CTXT_A
###  desc[3L] <- max(0, dim[1L])       # M_A
###  desc[4L] <- max(0, dim[2L])       # N_A
###  desc[5L] <- max(1, bldim[1L])     # MB_A
###  desc[6L] <- max(1, bldim[2L])     # NB_A
###  desc[7L] <- 0L                    # RSRC_A
###  desc[8L] <- 0L                    # CSRC_A
####  desc[9L] <- max(1L, ldim[1L])     # LLD_A
###  desc[9L] <- max(ldim[1L], max(1L, NUMROC(dim[1L], bldim[1L], grid$MYROW, grid$NPROW)))
  
  desc <- .Fortran("descinit", integer(9L), as.integer(dim[1L]), as.integer(dim[2L]), as.integer(bldim[1L]), as.integer(bldim[2L]), 0L, 0L, as.integer(ICTXT), as.integer(ldim[1L]), 0L)[[1L]]
  
  ### Fix for pdgemr2d: if a process is not a part of the given context, its ICTXT is -1
  if (any(base.blacs(ICTXT=ICTXT) == -1))
    desc[2L] <- -1L
  
  return(desc)
}

# R version of ScaLAPACK tool NUMROC
# Computes ldim of ddmatrix given dim and bldim
# if fixme=TRUE, then returned local dimensions which are < 1 will 
# be corrected (made = 1), and otherwise will be left unchanged.

#' @export
base.numroc <- function(dim, bldim, ICTXT=0, fixme=TRUE)
{
  
  blacs_ <- base.blacs(ICTXT=ICTXT)
  
  MYP <- c(blacs_$MYROW, blacs_$MYCOL)
  PROCS <- c(blacs_$NPROW, blacs_$NPCOL)
  
  ISRCPROC <- 0
  
  ldim <- numeric(2)
  for (i in 1:2){
    MYDIST <- (PROCS[i] + MYP[i] - ISRCPROC) %% PROCS[i]
    NBLOCKS <- floor(dim[i] / bldim[i])
    ldim[i] <- floor(NBLOCKS / PROCS[i]) * bldim[i]
    EXTRABLKS <- NBLOCKS %% PROCS[i]

    if (is.na(EXTRABLKS))
      EXTRABLKS <- 0

    if (MYDIST < EXTRABLKS)
      ldim[i] <- ldim[i] + bldim[i]
    else if (MYDIST == EXTRABLKS)
      ldim[i] <- ldim[i] + dim[i] %% bldim[i]
  }

  if (fixme){
    if (any(is.na(ldim)))
      ldim[which(is.na(ldim))] <- 0L
    if (any(ldim<1)) ldim <- c(1L, 1L) # FIXME
  }

  return(ldim)
}

numroc <- base.numroc


NUMROC <- function(N, NB, IPROC, NPROCS)
{
   ret <- .Call(R_NUMROC, as.integer(N), as.integer(NB), as.integer(IPROC), as.integer(NPROCS))
  
  return( ret )
}



#' Determining Local Ownership of a Distributed Matrix
#' 
#' For advanced users only.
#' 
#' \code{numroc()} is a re-implementation at the R level of the ScaLAPACK
#' subroutine NUMROC, which returns the local dimension of the matrix storage,
#' i.e. the dimension for the \code{Data} slot of the distributed matrix on
#' that process. The \code{fixme=} option, if \code{TRUE}, returns a minimum of
#' 1 for each dimension. If \code{fixme=FALSE}, then values less than 1 for
#' either dimension are possible, and in this case indicate a lack of local
#' ownership of the global matrix.
#' 
#' \code{ownany()} is a simple wrapper of numroc. The return is the answer to
#' the question 'do I own any of the global matrix?'.  Passing a distributed
#' matrix is allowed, but often it is convenient to determine that information
#' without even having a distributed matrix on hand. In this case, explicitly
#' passing the appropriate information to the arguments \code{dim=},
#' \code{bldim=} (and \code{CTXT=} as necessary, since it defaults to 0) while
#' leaving \code{x} missing will produce the desired result. See the examples
#' below for more clarity.
#' 
#' The return for each function is local.
#' 
#' @aliases Ownership base.ownany numroc
#' @param dim global dimension
#' @param bldim blocking dimension
#' @param ICTXT BLACS context
#' @param fixme logical, controls correction of local dimension return
#' 
#' @keywords BLACS Distributing Data
#' 
#' @examples
#' \dontrun{
#' # Save code in a file "demo.r" and run with 2 processors by
#' # > mpiexec -np 2 Rscript demo.r
#' 
#' library(pbdBASE, quiet = TRUE)
#' init.grid()
#' 
#' iown <- ownany(dim=c(4, 4), bldim=c(2, 2), CTXT=0)
#' comm.print(iown, all.rank=T)
#' 
#' finalize()
#' }
#' 
#' @export
base.ownany <- function(dim, bldim, ICTXT=0)
{
  if (length(bldim)==1)
    bldim <- rep(bldim, 2)
  
  grid <- base.blacs(ICTXT=ICTXT)
  
  check <- integer(2)
  
  check[1L] <- NUMROC(dim[1L], bldim[1L], grid$MYROW, grid$NPROW)
  check[2L] <- NUMROC(dim[2L], bldim[2L], grid$MYCOL, grid$NPCOL)
  
  if (any(check<1))
    return(FALSE)
  else
    return(TRUE)
}


# Hook into ScaLAPACK tool PDLAPRNT

#' @export
base.rpdlaprnt <- function(m, n, a, desca)
{
  if (!is.double(a))
    storage.mode(a) <- "double"
  
  .Call(R_PDLAPRNT, 
        as.integer(m), as.integer(n),
        a, as.integer(desca),
        as.character(deparse(substitute(a))),
        6L #WCC: 0 for stderr, 6 for stdout. Both are disabled.
        )
  
  return( invisible(0) )
}

# Compute maximum dimension across all nodes

#' @export
base.maxdim <- function(dim)
{
  mdim <- numeric(2)
  mdim[1] <- pbdMPI::allreduce(dim[1], op='max')
  mdim[2] <- pbdMPI::allreduce(dim[2], op='max')
  
  return( mdim )
}

# Compute dimensions on process MYROW=MYCOL=0

#' @export
base.dim0 <- function(dim, ICTXT=0)
{
  blacs_ <- base.blacs(ICTXT=ICTXT)
  MYROW <- blacs_$MYROW
  MYCOL <- blacs_$MYCOL
  
  if (MYROW == 0 && MYCOL == 0){
    mx01 <- dim[1]
    mx02 <- dim[2]
  }
  
  mx01 <- pbdMPI::bcast(mx01)
  mx02 <- pbdMPI::bcast(mx02)
  
#  pbdMPI::barrier()
  
  if (MYROW==0 && MYCOL==0)
    return( dim )
  else
    return( c(mx01, mx02) )
}



# l2g and g2l

#' @export
base.g2l_coord <- function(ind, dim, bldim, ICTXT=0)
{
  blacs_ <- base.blacs(ICTXT=ICTXT)
  procs <- c(blacs_$NPROW, blacs_$NPCOL)
  src <- c(0,0)
  
  out <- .Call(g2l_coords, 
                ind=as.integer(ind), dim=as.integer(dim), bldim=as.integer(bldim),
                procs=as.integer(procs), src=as.integer(src))
  
#  out[5:6] <- out[5:6] + 1
  
  if (out[3]!=blacs_$MYROW || out[4]!=blacs_$MYCOL)
    out <- rep(NA, 6)
  
  # out is a 'triple of pairs' stored as a length-6 vector, consisting of:
    # block position
    # process grid block
    # local coordinates
  # out will be a length 6 vector of NA when that global coord is not
  # relevant to the local storage
  
  return(out)
}

g2l_coord <- base.g2l_coord


#' @export
base.l2g_coord <- function(ind, dim, bldim, ICTXT=0)
{
  blacs_ <- base.blacs(ICTXT=ICTXT)
  procs <- c(blacs_$NPROW, blacs_$NPCOL)
  myproc <- c(blacs_$MYROW, blacs_$MYCOL)
  
  out <- .Call(l2g_coords, 
                ind=as.integer(ind), dim=as.integer(dim), bldim=as.integer(bldim),
                procs=as.integer(procs), src=as.integer(myproc))
  
  return(out)
}

l2g_coord <- base.l2g_coord