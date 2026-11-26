"""Outcome codes shared by the numerical methods. A method reports failure with one of these codes, never by raising, so that it can
run inside compiled code and on thousands of parameter values at once."""

import enum


class Status(enum.IntEnum):
    OK = 0
    NO_STABLE_SOLUTION = 1      # every solution explodes
    MANY_STABLE_SOLUTIONS = 2   # more than one solution does not explode, so the model does not pin down an answer
    NOT_CONVERGED = 3           # the iteration did not settle or produced non-finite numbers; or, for the forward-looking solve,
                                # the roots split correctly but no solution can be built from them (a "rank failure")
    SINGULAR = 4                # a matrix that must be invertible is not
