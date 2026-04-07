import os
import numpy as np
import pandas as pd
import opensim as osim
from .utilities import MultivariateNormal


def compute_measurement(model, state, station1, station2, dimension):
    station1 = osim.Station.safeDownCast(model.getComponent(f'/{station1}'))
    station2 = osim.Station.safeDownCast(model.getComponent(f'/{station2}'))
    pos1 = station1.getLocationInGround(state).to_numpy()
    pos2 = station2.getLocationInGround(state).to_numpy()
    delta = pos2 - pos1

    if dimension == 'x':
        return abs(delta[0])
    elif dimension == 'y':
        return abs(delta[1])
    elif dimension == 'z':
        return abs(delta[2])
    elif dimension == 'norm':
        return np.linalg.norm(delta)
    else:
        raise ValueError("dimension must be 'x', 'y', 'z', or 'norm'")



def adjust_anthropometry(scaled_model_fpath, anthropometrics_fpath,
                         adjusted_model_fpath):

    # Load the scaled model.
    model = osim.Model(scaled_model_fpath)
    state = model.initSystem()

    # Compute anthropometric measurements from the model.
    measurements = {
        'biacromialbreadth':      ('acromion_r', 'acromion_l', 'norm'),
        'bicristalbreadth':       ('iliocrestale_r', 'iliocrestale_l', 'norm'),
        'bimalleolarbreadth':     ('lateral_malleolus_r', 'medial_malleolus_r', 'norm'),
        'footbreadthhorizontal':  ('mtp1_r', 'mtp5_r', 'z'),
        'footlength':             ('acropodion_r', 'pternion_r', 'x'),
        # 'headbreadth':            ('euryon_r', 'euryon_l', 'norm'),
        # 'headlength':             ('glabella', 'opisthocranion', 'norm'),
        'iliocristaleheight':     ('iliocrestale_r', 'mtp5_r', 'y'),
        'lateralmalleolusheight': ('lateral_malleolus_r', 'mtp5_r', 'y'),
        'radialestylionlength':   ('radiale_r', 'stylion_r', 'norm'),
        'shoulderelbowlength':    ('acromion_r', 'olecranon_r', 'norm'),
        'stature':                ('vertex', 'mtp5_r', 'y'),
        'suprasternaleheight':    ('suprasternale', 'mtp5_r', 'y'),
        'tibialheight':           ('tibiale_r', 'mtp5_r', 'y'),
        'trochanterionheight':    ('trochanterion_r', 'mtp5_r', 'y'),
        'waistbacklength':        ('cervicale', 'posterior_omphalion', 'norm'),
        'waistdepth':             ('posterior_omphalion', 'anterior_omphalion', 'norm')
    }
    values = dict()
    for k, v in measurements.items():
        # Convert to millimeters to match the ANSUR II dataset.
        values[k] = 1000.0*compute_measurement(model, state, v[0], v[1], v[2])

    # Construct the multivariate normal distribution from the ANSUR II dataset. Use only
    # the measurements that we compute from the model.
    df = pd.read_csv(anthropometrics_fpath)
    df = df[measurements.keys()]
    mvn = MultivariateNormal.from_data(df.columns.tolist(), df.values)

    # Select of subset of the measurements that we will use to condition the
    # multivariate normal distribution. These measurements are "trustworthy" in the
    # sense that we can estimate them relatively well from the Theia frames.
    condition_values = {'iliocristaleheight': values['iliocristaleheight'],
                        'radialestylionlength': values['radialestylionlength'],
                        'shoulderelbowlength': values['shoulderelbowlength'],
                        'stature': values['stature'],
                        'suprasternaleheight': values['suprasternaleheight'],
                        'tibialheight': values['tibialheight'],
                        'trochanterionheight': values['trochanterionheight'],
                        'waistbacklength': values['waistbacklength']}

    # Condition the multivariate normal distribution on the selected measurements.
    mvn_conditioned = mvn.condition(condition_values)

    # Create a new dictionary with the conditioned measurement values.
    variables_conditioned = mvn_conditioned.get_variables()
    mean_conditioned = mvn_conditioned.get_mean()
    values_conditioned = dict()
    for var, mean in zip(variables_conditioned, mean_conditioned):
        values_conditioned[var] = mean

    # Create a new scale set to adjust based on the conditioned anthropometrics.
    scaleset = osim.ScaleSet()

    # torso
    scale = osim.Scale()
    scale.setSegmentName('torso')
    factors = osim.Vec3(1.0)
    factors[0] = values_conditioned['waistdepth'] / values['waistdepth']
    factors[2] = values_conditioned['biacromialbreadth'] / values['biacromialbreadth']
    scale.setScaleFactors(factors)
    scaleset.cloneAndAppend(scale)

    # pelvis
    scale = osim.Scale()
    scale.setSegmentName('pelvis')
    factors = osim.Vec3(1.0)
    factors[2] = values_conditioned['bicristalbreadth'] / values['bicristalbreadth']
    scale.setScaleFactors(factors)
    scaleset.cloneAndAppend(scale)

    # tibia
    scale = osim.Scale()
    scale.setSegmentName('tibia_r')
    factors = osim.Vec3(1.0)
    factors[0] = values_conditioned['bimalleolarbreadth'] / values['bimalleolarbreadth']
    factors[2] = values_conditioned['bimalleolarbreadth'] / values['bimalleolarbreadth']
    scale.setScaleFactors(factors)
    scaleset.cloneAndAppend(scale)

    scale = osim.Scale()
    scale.setSegmentName('tibia_l')
    factors = osim.Vec3(1.0)
    factors[0] = values_conditioned['bimalleolarbreadth'] / values['bimalleolarbreadth']
    factors[2] = values_conditioned['bimalleolarbreadth'] / values['bimalleolarbreadth']
    scale.setScaleFactors(factors)
    scaleset.cloneAndAppend(scale)

    # foot
    scale = osim.Scale()
    scale.setSegmentName('calcn_r')
    factors = osim.Vec3(1.0)
    factors[0] = values_conditioned['footlength'] / values['footlength']
    factors[1] = values_conditioned['lateralmalleolusheight'] / values['lateralmalleolusheight']
    factors[2] = values_conditioned['footbreadthhorizontal'] / values['footbreadthhorizontal']
    scale.setScaleFactors(factors)

    scale = osim.Scale()
    scale.setSegmentName('calcn_l')
    factors = osim.Vec3(1.0)
    factors[0] = values_conditioned['footlength'] / values['footlength']
    factors[1] = values_conditioned['lateralmalleolusheight'] / values['lateralmalleolusheight']
    factors[2] = values_conditioned['footbreadthhorizontal'] / values['footbreadthhorizontal']
    scale.setScaleFactors(factors)

    # Scale the model
    model.scale(state, scaleset, True)
    model.finalizeConnections()
    model.initSystem()
    model.printToXML(adjusted_model_fpath)
