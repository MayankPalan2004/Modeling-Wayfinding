import numpy as np
import matplotlib.pyplot as plt

params = {
    'k_tca': 0.1,
    'k_ATP_production': 0.05,
    'k_ADP_phosphorylation': 0.1,
    'k_NADH_oxidation': 0.05,
    'k_FADH2_oxidation': 0.05,
    'k_Ca2_uptake': 0.01,
    'k_Ca2_release': 0.01,
    'k_er_Ca2_transfer': 0.01,
    'k_cytochrome_c_release': 0.05,
    'k_caspase_activation': 0.1,
    'k_mtdna_replication': 0.05,
    'k_mtdna_degradation': 0.01,
    'k_ros_production_I': 0.02,
    'k_ros_production_III': 0.02,
    'k_ros_scavenging': 0.01,
    'k_biogenesis': 0.05,
    'k_mitophagy': 0.05,
    'k_glycolysis': 0.1,
    'k_fatty_acid_oxidation': 0.05,
    'k_amino_acid_metabolism': 0.05,
    'k_motility': 0.01,
    'k_fusion': 0.01,
    'k_fission': 0.01
}

def initialize_mitochondria(num_mitochondria, params):
    mitochondria = []
    for _ in range(num_mitochondria):
        mito = {
            'ΔΨm': np.random.normal(-150, 10),  
            'ATP': np.random.normal(2.0, 0.5),
            'ADP': np.random.normal(1.0, 0.2),
            'NADH': np.random.normal(1.0, 0.2),
            'NAD': np.random.normal(0.5, 0.1),
            'FADH2': np.random.normal(1.0, 0.2),
            'FAD': np.random.normal(0.5, 0.1),
            'Ca2+': np.random.normal(0.01, 0.005),
            'ROS': np.random.normal(0.01, 0.005),
            'cytochrome_c': 0.0,
            'caspase': 0.0,
            'mtDNA': np.random.normal(1.0, 0.2),
            'TFAM': np.random.normal(0.1, 0.02),
            'PGC1a': 0.0,
            'position': np.random.rand(2),
            'ubiquinone': np.random.normal(1.0, 0.2),
            'cytochrome_c_ox': np.random.normal(1.0, 0.2),
            'cytochrome_c_red': np.random.normal(0.0, 0.1)
        }
        mitochondria.append(mito)
    return mitochondria

mitochondria = initialize_mitochondria(100, params)

# TCA cycle function
def tca_cycle(mito, params):
    dATP = params['k_tca'] * mito['ADP']
    dNADH = params['k_tca'] * mito['NAD']
    dFADH2 = params['k_tca'] * mito['FAD']
    mito['ATP'] += dATP
    mito['ADP'] -= dATP
    mito['NADH'] += dNADH
    mito['NAD'] -= dNADH
    mito['FADH2'] += dFADH2
    mito['FAD'] -= dFADH2

# Oxidative phosphorylation function
def oxidative_phosphorylation(mito, params):
    dATP = params['k_ATP_production'] * mito['ADP']
    dNADH = params['k_NADH_oxidation'] * mito['NADH']
    dFADH2 = params['k_FADH2_oxidation'] * mito['FADH2']
    mito['ATP'] += dATP
    mito['ADP'] -= dATP
    mito['NADH'] -= dNADH
    mito['NAD'] += dNADH
    mito['FADH2'] -= dFADH2
    mito['FAD'] += dFADH2

# Calcium handling function
def calcium_handling(mito, params):
    dCa2_uptake = params['k_Ca2_uptake'] * mito['Ca2+']
    dCa2_release = params['k_Ca2_release'] * mito['Ca2+']
    mito['Ca2+'] += dCa2_uptake - dCa2_release

# ER calcium interaction function
def er_calcium_interaction(mito, er, params):
    dCa2_transfer = params['k_er_Ca2_transfer'] * (er['Ca2+'] - mito['Ca2+'])
    mito['Ca2+'] += dCa2_transfer
    er['Ca2+'] -= dCa2_transfer

# Apoptosis function
def apoptosis(mito, params):
    if mito['cytochrome_c'] > 0.1:
        dCaspase = params['k_caspase_activation'] * mito['cytochrome_c']
        mito['caspase'] += dCaspase
    if mito['caspase'] > 0.1:
        mito['caspase'] += params['k_cytochrome_c_release'] * mito['caspase']
        mito['cytochrome_c'] = 0

# mtDNA dynamics function
def mtDNA_dynamics(mito, params):
    dmtDNA_replication = params['k_mtdna_replication'] * mito['TFAM']
    dmtDNA_degradation = params['k_mtdna_degradation'] * mito['ROS']
    mito['mtDNA'] += dmtDNA_replication - dmtDNA_degradation

# ROS dynamics function
def ros_dynamics(mito, params):
    dROS_production_I = params['k_ros_production_I'] * mito['NADH']
    dROS_production_III = params['k_ros_production_III'] * mito['FADH2']
    dROS_scavenging = params['k_ros_scavenging'] * mito['ATP']
    mito['ROS'] += dROS_production_I + dROS_production_III - dROS_scavenging

# Biogenesis and turnover function
def biogenesis_and_turnover(mito, params):
    dPGC1a_dt = params['k_biogenesis'] * mito['ATP']
    dMitophagy_dt = params['k_mitophagy'] * mito['ROS']
    mito['PGC1a'] += dPGC1a_dt
    mito['caspase'] += dMitophagy_dt

# Cellular metabolism function
def cellular_metabolism(mito, params):
    dglycolysis_dt = params['k_glycolysis'] * mito['ADP']
    dfatty_acid_oxidation_dt = params['k_fatty_acid_oxidation'] * mito['ATP']
    damino_acid_metabolism_dt = params['k_amino_acid_metabolism'] * mito['ATP']
    mito['ATP'] += (dglycolysis_dt + dfatty_acid_oxidation_dt + damino_acid_metabolism_dt)
    mito['ADP'] -= (dglycolysis_dt + dfatty_acid_oxidation_dt + damino_acid_metabolism_dt)

# Mitochondrial motility function
def mitochondrial_motility(mito, params):
    displacement = params['k_motility'] * (np.random.rand(2) - 0.5)
    mito['position'] += displacement

# Fusion and fission function
def fusion_and_fission(mitochondria, params):
    if np.random.rand() < params['k_fusion']:
        idx1, idx2 = np.random.choice(len(mitochondria), 2)
        if np.linalg.norm(mitochondria[idx1]['position'] - mitochondria[idx2]['position']) < 0.1:
            new_mito = {
                'ΔΨm': (mitochondria[idx1]['ΔΨm'] + mitochondria[idx2]['ΔΨm']) / 2,
                'ATP': (mitochondria[idx1]['ATP'] + mitochondria[idx2]['ATP']) / 2,
                'ADP': (mitochondria[idx1]['ADP'] + mitochondria[idx2]['ADP']) / 2,
                'NADH': (mitochondria[idx1]['NADH'] + mitochondria[idx2]['NADH']) / 2,
                'NAD': (mitochondria[idx1]['NAD'] + mitochondria[idx2]['NAD']) / 2,
                'FADH2': (mitochondria[idx1]['FADH2'] + mitochondria[idx2]['FADH2']) / 2,
                'FAD': (mitochondria[idx1]['FAD'] + mitochondria[idx2]['FAD']) / 2,
                'Ca2+': (mitochondria[idx1]['Ca2+'] + mitochondria[idx2]['Ca2+']) / 2,
                'ROS': (mitochondria[idx1]['ROS'] + mitochondria[idx2]['ROS']) / 2,
                'cytochrome_c': (mitochondria[idx1]['cytochrome_c'] + mitochondria[idx2]['cytochrome_c']) / 2,
                'caspase': (mitochondria[idx1]['caspase'] + mitochondria[idx2]['caspase']) / 2,
                'mtDNA': (mitochondria[idx1]['mtDNA'] + mitochondria[idx2]['mtDNA']) / 2,
                'TFAM': (mitochondria[idx1]['TFAM'] + mitochondria[idx2]['TFAM']) / 2,
                'PGC1a': (mitochondria[idx1]['PGC1a'] + mitochondria[idx2]['PGC1a']) / 2,
                'position': (mitochondria[idx1]['position'] + mitochondria[idx2]['position']) / 2,
                'ubiquinone': (mitochondria[idx1]['ubiquinone'] + mitochondria[idx2]['ubiquinone']) / 2,
                'cytochrome_c_ox': (mitochondria[idx1]['cytochrome_c_ox'] + mitochondria[idx2]['cytochrome_c_ox']) / 2,
                'cytochrome_c_red': (mitochondria[idx1]['cytochrome_c_red'] + mitochondria[idx2]['cytochrome_c_red']) / 2
            }
            mitochondria.append(new_mito)
            mitochondria.pop(idx1)
            mitochondria.pop(idx2-1)
    if np.random.rand() < params['k_fission']:
        idx = np.random.choice(len(mitochondria))
        new_mito1 = mitochondria[idx].copy()
        new_mito2 = mitochondria[idx].copy()
        displacement = params['k_motility'] * (np.random.rand(2) - 0.5)
        new_mito1['position'] += displacement
        new_mito2['position'] -= displacement
        mitochondria.append(new_mito1)
        mitochondria.append(new_mito2)
        mitochondria.pop(idx)

def simulate_mitochondria(num_steps, params):
    global mitochondria
    for _ in range(num_steps):
        for mito in mitochondria:
            tca_cycle(mito, params)
            oxidative_phosphorylation(mito, params)
            calcium_handling(mito, params)
            mtDNA_dynamics(mito, params)
            ros_dynamics(mito, params)
            biogenesis_and_turnover(mito, params)
            cellular_metabolism(mito, params)
            mitochondrial_motility(mito, params)
        fusion_and_fission(mitochondria, params)

def visualize_mitochondria():
    positions = np.array([mito['position'] for mito in mitochondria])
    atp_levels = np.array([mito['ATP'] for mito in mitochondria])
    plt.scatter(positions[:, 0], positions[:, 1], c=atp_levels, cmap='viridis')
    plt.colorbar(label='ATP Levels')
    plt.xlabel('Position X')
    plt.ylabel('Position Y')
    plt.title('Mitochondrial ATP Levels and Positions')
    plt.show()


simulate_mitochondria(num_steps=1000, params=params)
visualize_mitochondria()

