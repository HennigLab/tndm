import argparse, sys, os
import pytest

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches

from latentneural.legacy.psid import LSSM, PSID
from latentneural.legacy.psid.evaluation import evalPrediction
from latentneural.legacy.psid.MatHelper import loadmat


@pytest.fixture(scope='module')
def tutorial_input_data():
    # Load data
    sample_model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data', 'sample_model.mat')
                    
    print('Loading example model from {}'.format(sample_model_path))
    data = loadmat(sample_model_path)

    # Generating some sample data from this model
    np.random.seed(42) # For exact reproducibility

    N = int(2e4)
    trueSys = LSSM(params=data['trueSys'])
    y, x = trueSys.generateRealization(N)
    z = (trueSys.Cz @ x.T).T

    # Add some z dynamics that are not encoded in y (i.e. epsilon)
    epsSys = LSSM(params=data['epsSys'])
    eps, _ = epsSys.generateRealization(N)
    z += eps

    allYData, allZData = y, z

    # Separate data into training and test data:
    trainInds = np.arange(np.round(0.5*allYData.shape[0]), dtype=int)
    testInds = np.arange(1+trainInds[-1], allYData.shape[0])
    yTrain = allYData[trainInds, :]
    yTest = allYData[testInds, :]
    zTrain = allZData[trainInds, :]
    zTest = allZData[testInds, :]

    return (data, trueSys, allYData, allZData, yTrain, yTest, zTrain, zTest)


@pytest.mark.smoke
@pytest.mark.legacy
def test_tutorial_with_generated_data(tutorial_input_data):
    (data, trueSys, allYData, allZData, yTrain, yTest, zTrain, zTest) = tutorial_input_data

    ## (Example 1) PSID can be used to dissociate and extract only the 
    # behaviorally relevant latent states (with nx = n1 = 2)
    idSys1 = PSID(yTrain, zTrain, nx=2, n1=2, i=10)

    # Predict behavior using the learned model
    zTestPred1, yTestPred1, xTestPred1 = idSys1.predict(yTest)

    # Compute CC of decoding
    CC = evalPrediction(zTest, zTestPred1, 'CC')

    # Predict behavior using the true model for comparison
    zTestPredIdeal, yTestPredIdeal, xTestPredIdeal = trueSys.predict(yTest)
    CCIdeal = evalPrediction(zTest, zTestPredIdeal, 'CC')

    print('Behavior decoding CC:\n  PSID => {:.3g}, Ideal using true model => {:.3g}'.format(np.mean(CC), np.mean(CCIdeal)) )

    ## (Example 2) Optionally, PSID can additionally also learn the 
    # behaviorally irrelevant latent states (with nx = 4, n1 = 2)
    idSys2 = PSID(yTrain, zTrain, nx=4, n1=2, i=10)

    # In addition to ideal behavior decoding, this model will also have ideal neural self-prediction 
    zTestPred2, yTestPred2, xTestPred2 = idSys2.predict(yTest)
    yCC2 = evalPrediction(yTest, yTestPred2, 'CC')
    yCCIdeal = evalPrediction(yTest, yTestPredIdeal, 'CC')
    print('Neural self-prediction CC:\n  PSID => {:.3g}, Ideal using true model => {:.3g}'.format(np.mean(yCC2), np.mean(yCCIdeal)))

    # #########################################
    # Plot the true and identified eigenvalues    

    # (Example 1) Eigenvalues when only learning behaviorally relevant states
    idEigs1 = np.linalg.eig(idSys1.A)[0]

    # (Example 2) Additional eigenvalues when also learning behaviorally irrelevant states
    # The identified model is already in form of Eq. 4, with behaviorally irrelevant states 
    # coming as the last 2 dimensions of the states in the identified model
    idEigs2 = np.linalg.eig(idSys2.A[2:, 2:])[0]

    relevantDims = trueSys.zDims - 1 # Dimensions that drive both behavior and neural activity
    irrelevantDims = [x for x in np.arange(trueSys.state_dim, dtype=int) if x not in relevantDims] # Dimensions that only drive the neural activity
    trueEigsRelevant = np.linalg.eig(trueSys.A[np.ix_(relevantDims, relevantDims)])[0]
    trueEigsIrrelevant = np.linalg.eig(trueSys.A[np.ix_(irrelevantDims, irrelevantDims)])[0]
    nonEncodedEigs = np.linalg.eig(data['epsSys']['a'])[0] # Eigenvalues for states that only drive behavior

    fig = plt.figure(figsize=(8, 4))
    axs = fig.subplots(1, 2)
    axs[1].remove() 
    ax = axs[0]
    ax.axis('equal')
    ax.add_patch( patches.Circle((0,0), radius=1, fill=False, color='black', alpha=0.2, ls='-') )
    ax.plot([-1,1,0,0,0], [0,0,0,-1,1], color='black', alpha=0.2, ls='-')
    ax.scatter(np.real(nonEncodedEigs), np.imag(nonEncodedEigs), marker='o', edgecolors='#0000ff', facecolors='none', label='Not encoded in neural signals')
    ax.scatter(np.real(trueEigsIrrelevant), np.imag(trueEigsIrrelevant), marker='o', edgecolors='#ff0000', facecolors='none', label='Behaviorally irrelevant')
    ax.scatter(np.real(trueEigsRelevant), np.imag(trueEigsRelevant), marker='o', edgecolors='#00ff00', facecolors='none', label='Behaviorally relevant')
    ax.scatter(np.real(idEigs1), np.imag(idEigs1), marker='x', facecolors='#00aa00', label='PSID Identified (stage 1)')
    ax.scatter(np.real(idEigs2), np.imag(idEigs2), marker='x', facecolors='#aa0000', label='(optional) PSID Identified (stage 2)')
    ax.set_title('True and identified eigevalues')
    ax.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
    # plt.show()

@pytest.mark.smoke
def test_tutorial_with_trial_data(tutorial_input_data):
    (data, trueSys, allYData, allZData, yTrain, yTest, zTrain, zTest) = tutorial_input_data

    ## (Example 3) PSID can be used if data is available in discontinuous segments (e.g. different trials)
    # In this case, y and z data segments must be provided as elements of a list
    # Trials do not need to have the same number of samples
    # Here, for example assume that trials start at every 1000 samples.
    # And each each trial has a random length of 500 to 900 samples
    trialStartInds = np.arange(0, allYData.shape[0]-1000, 1000)
    trialDurRange = np.array([900, 990])
    trialDur = np.random.randint(low=trialDurRange[0], high=1+trialDurRange[1], size=trialStartInds.shape)
    trialInds = [trialStartInds[ti]+np.arange(trialDur[ti]) for ti in range(trialStartInds.size)] 
    yTrials = [allYData[trialIndsThis, :] for trialIndsThis in trialInds] 
    zTrials = [allZData[trialIndsThis, :] for trialIndsThis in trialInds] 

    # Separate data into training and test data:
    trainInds = np.arange(np.round(0.5*len(yTrials)), dtype=int)
    testInds = np.arange(1+trainInds[-1], len(yTrials))
    yTrain = [yTrials[ti] for ti in trainInds]
    yTest = [yTrials[ti] for ti in testInds]
    zTrain = [zTrials[ti] for ti in trainInds]
    zTest = [zTrials[ti] for ti in testInds]

    idSys3 = PSID(yTrain, zTrain, nx=2, n1=2, i=10)

    for ti in range(len(yTest)):
        zPredThis, yPredThis, xPredThis = idSys3.predict(yTest[ti])
        zPredThisIdeal, yPredThisIdeal, xPredThisIdeal = trueSys.predict(yTest[ti])
        if ti == 0:
            zTestA = zTest[ti]
            zPredA = zPredThis
            zPredIdealA = zPredThisIdeal
        else:
            zTestA = np.concatenate( (zTestA, zTest[ti]), axis=0)
            zPredA = np.concatenate( (zPredA, zPredThis), axis=0)
            zPredIdealA = np.concatenate( (zPredIdealA, zPredThisIdeal), axis=0)

    CCTrialBased = evalPrediction(zTestA, zPredA, 'CC')
    CCTrialBasedIdeal = evalPrediction(zTestA, zPredIdealA, 'CC')

    print('Behavior decoding CC (trial-based learning/decoding):\n  PSID => {:.3g}, Ideal using true model = {:.3g}'.format(np.mean(CCTrialBased), np.mean(CCTrialBasedIdeal)) )