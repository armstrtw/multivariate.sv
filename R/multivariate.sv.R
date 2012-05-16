###########################################################################
## Copyright (C) 2012  Whit Armstrong                                    ##
##                                                                       ##
## This program is free software: you can redistribute it and/or modify  ##
## it under the terms of the GNU General Public License as published by  ##
## the Free Software Foundation, either version 3 of the License, or     ##
## (at your option) any later version.                                   ##
##                                                                       ##
## This program is distributed in the hope that it will be useful,       ##
## but WITHOUT ANY WARRANTY; without even the implied warranty of        ##
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         ##
## GNU General Public License for more details.                          ##
##                                                                       ##
## You should have received a copy of the GNU General Public License     ##
## along with this program.  If not, see <http:##www.gnu.org/licenses/>. ##
###########################################################################


multivariate.sv <- function(X,iterations,burn,adapt,thin) {

    if(typeof(X)!="double") {
        stop("data must be double.")
    }

    if(!("matrix" %in% class(X))) {
        stop("X must be a matrix.")
    }

    ans <- .Call("multivariate_sv", X, as.integer(iterations), as.integer(burn), as.integer(adapt), as.integer(thin), PACKAGE="multivariate.sv")

    ans
}
