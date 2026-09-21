"""Fixed two-sided terminal-loss decision; epsilon is supplied, never optimized here."""
import math

def select(loss0, loss_plus, loss_minus, epsilon, zero_direction=False):
    values=(loss0,loss_plus,loss_minus,epsilon)
    if any(v is None or not math.isfinite(v) for v in values) or epsilon<0:
        raise ValueError('missing/nonfinite probe loss or invalid epsilon')
    if epsilon==0 and not zero_direction:raise ValueError('zero epsilon without ZERO_DIRECTION')
    tau=1e-6*max(1.,abs(loss0),abs(loss_plus),abs(loss_minus))
    plus=loss0-loss_plus>tau;minus=loss0-loss_minus>tau
    if zero_direction:
        if epsilon!=0:raise ValueError('ZERO_DIRECTION requires epsilon zero')
        sign=0;status='ZERO_DIRECTION'
    elif not plus and not minus:sign=0;status='SKIP_NONIMPROVING'
    elif plus and not minus:sign=1;status='SELECT_PLUS'
    elif minus and not plus:sign=-1;status='SELECT_MINUS'
    elif abs(loss_plus-loss_minus)<=tau:sign=1;status='SELECT_PLUS_TIE'
    elif loss_plus<loss_minus:sign=1;status='SELECT_PLUS'
    else:sign=-1;status='SELECT_MINUS'
    return dict(status=status,sign=sign,epsilon=epsilon,tau=tau,loss0=loss0,loss_plus=loss_plus,loss_minus=loss_minus,
                eligible_plus=plus,eligible_minus=minus,
                central_difference=None if epsilon==0 else (loss_plus-loss_minus)/(2*epsilon),
                nonlinear_symmetric_difference=loss_plus+loss_minus-2*loss0,
                tau_meaning='numerical decision deadband only, not effect threshold or stochastic-noise guarantee',
                derivative_meaning='finite-amplitude secant slope in unit-clean direction; not a local-derivative guarantee or step-size rule')
