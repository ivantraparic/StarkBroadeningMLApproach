from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from sklearn.pipeline import Pipeline

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

from scipy.signal import find_peaks
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as py
import re
import math
import os

#---------------------------------------------------------------------- Class for finding available transitions in database -------------------------------------------
class TransitionFinding():

    def __init__(self, pathNIST, pathStark):
        self.pathNIST = pathNIST
        self.pathStark = pathStark
        
    def Get_File_Names(self, path):
        folder = os.fsencode(path) 
        filenames = []
        #geting Nist filenames
        for file in os.listdir(folder):
            filename = os.fsdecode(file)
            if filename.endswith('.txt'):
                filenames.append(filename)

        filenames.sort()

        return filenames

    def StarkConfigurations(self, lowerlevel, upperlevel):
        upperlevelconf = []
        lowerlevelconf = []
        upperlevelstate = []
        lowerlevelstate = []

        for k in range(len(lowerlevel)):
            upperspaceindex = upperlevel[k].index(" ")
            upperlevelconf.append(upperlevel[k][0:upperspaceindex])
            upperlevelstate.append(upperlevel[k][upperspaceindex+1])

            lowerspaceindex = lowerlevel[k].index(" ")
            lowerlevelconf.append(lowerlevel[k][0:lowerspaceindex])
            lowerlevelstate.append(lowerlevel[k][lowerspaceindex+1])

        return lowerlevelconf, upperlevelconf, lowerlevelstate, upperlevelstate

    def FindTransitions(self, emitername, Ne, T):
        emiter_charge = {'I':+0, 'II':+1, 'III':+2, 'IV':+3, 'V':+4, 'VI':+5, 'VII':+6, 'VIII':+7, 'IX':+8, 'X':+9, 'XI':+10}
        
        ind = emitername.index(" ")
        name = emitername[0:ind]
        charge = emiter_charge.get(emitername[ind+1])

        nistName = name+"+"+str(charge)

        starkName = name+"_"+emitername[ind+1]

        filenamesN = self.Get_File_Names(self.pathNIST)
        filenamesS = self.Get_File_Names(self.pathStark)
        

        for names in filenamesN:
            if names == "{}.txt".format(nistName):
                FileN = pd.read_csv('{}/{}'.format(self.pathNIST, names), delimiter = "\t", header = 0, engine = 'python')
                break
        
        for names in filenamesS:
            if (starkName in names) and (Ne in names):
                FileS = pd.read_csv('{}/{}'.format(self.pathStark, names), delimiter = "\t", header = 0, engine = 'python')
                break

        config_NIST = FileN["Configuration"].values.tolist()
        upperlevel = FileS["Upper Level"].values.tolist()
        lowerlevel = FileS["Lower Level"].values.tolist()
        electronW = FileS["electron w"].values.tolist()
        Temp = FileS["T(K)"].values.tolist()

        lowerlevelconf, upperlevelconf, lowerlevelstate, upperlevelstate = self.StarkConfigurations(lowerlevel, upperlevel)
        counter = 0
        length_NIST_config = [len(x) for x in config_NIST]
        TransitionTerm = []
        lower_level_cor = []
        upper_level_cor = []
        transitions = []
        while counter != len(lowerlevelconf):
            if Temp[counter] == T:
                #Taking the configuration in Stark b
                help_config_lower = lowerlevelconf[counter]
                help_config_upper = upperlevelconf[counter]
                #See if match exists
                #Lower level
                condition_lower = False

                for j in range(len(length_NIST_config)):
                    help_config_lower1 = help_config_lower[-length_NIST_config[j]:]
                    if help_config_lower1 == "":
                        continue
                    else:
                        for i in range(len(config_NIST)):
                            if help_config_lower1 == config_NIST[i]:
                                lower_level_cor.append(help_config_lower1)
                                condition_lower = True
                                TransitionTerm.append(lowerlevelstate[counter])
                                break
                        
                    if condition_lower == True:
                        break

                #Upper level
                condition_upper = False
                for k in range(len(length_NIST_config)):
                    help_config_upper1 = help_config_upper[-length_NIST_config[k]:]
                    if help_config_upper1 == "":
                        continue
                    else:
                        for m in range(len(config_NIST)):
                            if help_config_upper1 == config_NIST[m]:
                                upper_level_cor.append(help_config_upper1)
                                condition_upper = True
                                break
                        
                    if condition_upper == True:                   
                        break

                pointindexupper = upper_level_cor[-1].index(".")
                uplev = upper_level_cor[-1][pointindexupper+1:]

                pointindexlower = lower_level_cor[-1].index(".")
                downlev = lower_level_cor[-1][pointindexlower+1:]
            
                transitions.append(("{}-{}".format(downlev,uplev), electronW[counter]))
                counter += 1
            else:
                counter += 1

        return transitions, TransitionTerm
            
    def MatchingTransitions(self, transitions, wanted_transitions, terms):
        all_transitions = []
        wantedResult = []
        for j in range(len(wanted_transitions)):
            Result = False
            nlocation = wanted_transitions[j].index("n")
            for index, vector in enumerate(transitions):
                search_prompt1 = wanted_transitions[j][0:nlocation] + '[0-9]' + wanted_transitions[j][nlocation+1]
                search_prompt2 = wanted_transitions[j][0:nlocation] + '[1-9][0-9]' + wanted_transitions[j][nlocation+1]
                finds1 = re.findall(search_prompt1, vector[0])
                finds2 = re.findall(search_prompt2, vector[0])
                if len(finds1) != 0:
                    Result = True
                    all_transitions.append(((wanted_transitions[j]+"({})".format(terms[index]),vector[0]+"({})".format(terms[index])), vector[1]))
                if len(finds2) != 0:
                    Result = True
                    all_transitions.append(((wanted_transitions[j]+"({})".format(terms[index]),vector[0]+"({})".format(terms[index])), vector[1]))

            wantedResult.append(Result)
        return all_transitions, wantedResult

    def RemoveNans(self, transitions):
        
        for vector in reversed(transitions):
            if math.isnan(vector[1]) == True:
                transitions.remove(vector)
            
        return transitions
    
#----------------------------------------------------------------------- Clean data ----------------------------------------------------------------------------------
Data = pd.read_csv(r'C:\Users\ivant\Desktop\Python\MLFakultet\Databases\MLData.csv',delimiter = ',', header = 0, engine='python')
Data.columns = ["Emiter", "TD", "Temp", "Ne","chi", "w", "wrad", "Z", "Osn nivo", "Gornji nivo","J_gornje","Donji nivo", "J_donje","ni", "nf", "li", "lf"]
data = pd.DataFrame(Data)

df1 = data[~(data["Gornji nivo"]<data["Donji nivo"])]

df4 = df1.loc[df1['Gornji nivo']>500]
indexes_to_drop = df4.index
df5 = df1.drop(indexes_to_drop, axis = 0)

df6 = df5.loc[df5["Temp"]>150000]
indexes = df6.index
df7 = df5.drop(indexes, axis = 0)

df8 = df7.loc[df7['Ne']>1e18]
indexes_to_drop = df8.index
df10 = df7.drop(indexes_to_drop, axis = 0)

#----------------------------------------------------------------------- Training data and target variable ------------------------------------------------------------

#Target variable
y_array = df10["wrad"].values
        
X = df10.drop(["TD","w","wrad"], axis = 1) #droping wavelength, w in angstroms and w in radians per s
X_array = X.values #features

#----------------------------------------------------------------------------- Cross validation -----------------------------------------------------------------------
#Cross Validation check of best parameters for regressors
kf = ShuffleSplit(n_splits = 5, random_state = 0, test_size = 0.3)
scores = []

X_train, X_test, y_train, y_test = train_test_split(X_array, y_array, test_size = 0.25, shuffle = True)

models = {
     'Linear Regression':{
         'model': LinearRegression(),
         'params': {
             'model__normalize':[True, False]
         }
     },
    
     'DecisionTreeRegressor':{
         'model': DecisionTreeRegressor(),
         'params': {
             'model__max_depth':[3,5,10]
         }
        
     },
    
     'Random Forest Regressor':{
         'model': RandomForestRegressor(),
         'params':{
             'model__n_estimators' : [5,10,15,100]
         }
     },
    
     'GradientBoostingRegressor':{
         'model': GradientBoostingRegressor(),
         'params':{
             'model__max_depth':[3,5,10],
             'model__n_estimators':[100,150,200]
         }
     }
 }
       
for key_model, modelp in models.items():
    pipe = Pipeline(steps=[('standardize', StandardScaler()), ('model', modelp['model'])])
    
    clf = GridSearchCV(pipe, modelp['params'], cv = kf,return_train_score = False, n_jobs=-1)
    clf.fit(X_train, y_train)

    scores.append({'Model':key_model, 'Best Score':clf.best_score_, 'Best Params':clf.best_params_})

df = pd.DataFrame(scores, columns = ['Model', 'Best Score', 'Best Params'])
df.head()

#------------------------------------------------------------ Training and testing model performance ------------------------------------------------------------------

rf_model = Pipeline(steps = [('standardize', StandardScaler()),('model', RandomForestRegressor(n_estimators=100))])

#Model Training
rf_model.fit(X_train,y_train)


#R^2 showing - to check the overfitting

print(f'Training score: {rf_model.score(X_train, y_train)}')
print(f'Test score: {rf_model.score(X_test,y_test)}')


#-------------------------------------------------------- Preprocessing of data to predict ----------------------------------------------------------------------------
# We introduced additional columns Transition and Term later for drawing transitions on the graphs

data3 = pd.read_csv(r"C:\Users\ivant\Desktop\Python\MLFakultet\Li I 1e16 30000K\LiPredictionData.csv", delimiter = ',', header = 0) #data to predict Stark width

OldData = False #If we previously investigated this particular element, to compare results with ML model

prompt = input("Do you have old predictions in input data (y/n). If yes, enter the name of column: ")

if prompt != 'n' or prompt != 'N':
    OldData = True

if OldData == False:
    data3.columns = ["Emiter", "TD","Temp", "Ne","chi","Z", "Osn nivo", "Gornji nivo","J_gornje","Donji nivo", "J_donje", "ni", "nf", "li", "lf","Transition","Term"]
    data3 = pd.DataFrame(data3)


else:
    data3.columns = ["Emiter", "TD","Temp", "Ne","chi","Z", "Osn nivo", "Gornji nivo","J_gornje","Donji nivo", "J_donje", "ni", "nf", "li", "lf","w1","w2","{}".format(prompt),"w2r","Transition","Term"]
    data3 = pd.DataFrame(data3)
    w_old = data3["{}".format(prompt)].values.tolist()

    zero_index = []
    for k in range(len(w_old)):
        if w_old[k] == 0:
            zero_index.append(True)
        else:
            zero_index.append(False)

    
multiplet = {'1':'singlet', '2':'doublet', '3':'triplet', '4':'quartet', '5':'quintet', '6':'sextet', '7':'septet'}

#Declaring variables
term = data3["Term"].values.tolist()

Transition = data3["Transition"].values.tolist()
wavelen = data3["TD"].values.tolist()

print("Independent variable to plot(Emiter, Temp, Ne, chi, Z, Osn nivo, Gornji nivo, ni, nf, li, lf, Donji nivo, Nepoznato)")
indp_var = input("Choice: ")
unit = input("Enter independent variable unit: ")
x_plot = data3[indp_var].values.tolist() # Value to plot 
temp = input("Enter Temperature: ")

#Geting all multiplets from dataset
DiferentTerms = []
for i in range(len(term)):
    if term[i] not in DiferentTerms:
        DiferentTerms.append(term[i])

if OldData == False:
    data3 = data3.drop(["TD","Transition","Term"],axis = 1)
else:
    data3 = data3.drop(["TD","Transition","Term","w1","w2","{}".format(prompt),"w2r"],axis = 1) 

#Separaing data to multiplets
if len(DiferentTerms) == 1:    
    Xpredict = data3.values    
    TransitionName = ["{}({})".format(Transition[i],DiferentTerms[0]) for i in range(len(Transition))]

else:
    WavelenList = [[] for j in range(len(DiferentTerms))]
    TransitionList = [[] for j in range(len(DiferentTerms))]
    Indexes = [[] for j in range(len(DiferentTerms))]
    Xpredict = [[] for j in range(len(DiferentTerms))]
    x = [[] for j in range(len(DiferentTerms))]

    if OldData == True:
        WOld = [[] for j in range(len(DiferentTerms))]
        for j in range(len(WOld)):
            WOld[j] = [w_old[k] for k in range(len(w_old)) if term[k] == DiferentTerms[j]]

    for j in range(len(DiferentTerms)):
        WavelenList[j] = [wavelen[k] for k in range(len(wavelen)) if term[k] == DiferentTerms[j]]
        TransitionList[j] = ["{}({})".format(Transition[k], DiferentTerms[j]) for k in range(len(Transition)) if term[k] == DiferentTerms[j]]
        Indexes[j] = [k for k in range(len(term)) if term[k] == DiferentTerms[j]]
        Xpredict[j] = (data3.loc[Indexes[j],:]).values
        
        x[j] = [x_plot[k] for k in range(len(x_plot)) if term[k] == DiferentTerms[j]]

#-------------------------------------------------------------------------- If we want to fix certain transitions -----------------------------------------------------

c = input("Do you want to fix transitions (y/n): ")
#If answer is positive
if c == 'y':
    t = input("Enter transitions: ")
    if ',' in t:
        tran = t.split(',')
        tran = [tran.strip() for tran in tran]
    else:
        tran = list(t)

    #If we have only one multiplet in dataset
    if len(DiferentTerms) == 1: 

        transition_fix = ["{}({})".format(Transition[i],term[i]) for i in range(len(Transition)) if Transition[i] in tran]

        IndexesFix = [k for k in range(len(Transition)) if Transition[k] in tran]
        Xpredict_fix = data3[IndexesFix,:].values
        
        x_fix = [x_plot[i] for i in range(len(x_plot)) if Transition[i] in tran]
                
        if OldData == True:
            wr_fix = [w_old[i] for i in range(len(w_old)) if Transition[i] in tran]

        wavelen_fix = [wavelen[i] for i in range(len(wavelen)) if Transition[i] in tran]

        #Variables for ploting
            
        log_chi = [np.log10(x) for x in x_fix]
        old_log_chi = [np.log10(x) for x in x_fix if zero_index[x_plot.index(x)] != True]

        if OldData == True:
            old_log = [np.log10(x) for x in wr_fix if zero_index[wr_fix.index(x)] != True]
    
    #If we have more than one multiplet
    else:
        transition_fix = [[] for i in range(len(DiferentTerms))]
        x_fix = [[] for i in range(len(DiferentTerms))]
        wavelen_fix = [[] for i in range(len(DiferentTerms))]
        log_chi = [[] for i in range(len(DiferentTerms))]
        Xpredict_fix = [[] for i in range(len(DiferentTerms))]
        IndexesFix = [[] for i in range(len(DiferentTerms))]
        old_log_chi = [[] for i in range(len(DiferentTerms))]

        if OldData == True:
            wr_fix = [[] for i in range(len(DiferentTerms))]
            old_log = [[] for i in range(len(DiferentTerms))]
        
        for j in range(len(DiferentTerms)):
            transition_fix[j] = ["{}({})".format(Transition[i], term[i]) for i in range(len(Transition)) if Transition[i] in tran and DiferentTerms[j] == term[i]]
            x_fix[j] = [x_plot[i] for i in range(len(x_plot)) if Transition[i] in tran and DiferentTerms[j] == term[i]]
            wavelen_fix[j] = [wavelen[i] for i in range(len(wavelen)) if Transition[i] in tran and DiferentTerms[j] == term[i]]
            IndexesFix[j] = [i for i in range(len(Transition)) if Transition[i] in tran and DiferentTerms[j] == term[i]]
            Xpredict_fix[j] = data3.loc[IndexesFix[j],:].values
            log_chi[j] = [np.log10(x_fix[j][i]) for i in range(len(x_fix[j]))]

            if OldData == True:
                wr_fix[j] = [w_old[i] for i in range(len(w_old)) if Transition[i] in tran and DiferentTerms[j] == term[i] and zero_index[w_old.index(w_old[i])] != True]
                old_log[j] = [np.log10(wr_fix[j][i]) for i in range(len(wr_fix[j]))]
                old_log_chi[j] = [np.log10(x1) for x1 in x_fix[j] if zero_index[x_plot.index(x1)] != True]

#If answer is negative
else:
    if len(DiferentTerms) == 1: 
            
        log_chi = [np.log10(x) for x in x_plot]

        if OldData == True:
            old_log = [np.log10(x) for x in w_old]
            old_log_chi = [np.log10(x) for x in x_plot if zero_index[x_plot.index(x)] != True]

    else:
        
        log_chi = [[] for i in range(len(DiferentTerms))]

        if OldData == True:
            old_log = [[] for i in range(len(DiferentTerms))]
            old_log_chi = [[] for i in range(len(DiferentTerms))]
        
        for j in range(len(DiferentTerms)):
            log_chi[j] = [np.log10(x[j][i]) for i in range(len(x[j]))] 

            if OldData == True:
                old_log[j] = [np.log10(WOld[j][i]) for i in range(len(WOld[j])) if zero_index[WOld[j].index(WOld[j][i])] != True]
                old_log_chi = [np.log10(x) for x in x[j] if zero_index[x_plot.index(x)] != True]

#--------------------------------------------------------------------- Preditctions with wining model and drawing graphs -----------------------------------------------------------------
el_w = []
multiplet = {'1':'singlet', '2':'doublet', '3':'triplet', '4':'quartet', '5':'quintet', '6':'sextet', '7':'septet'}
c = 'n'
OldData = False
indp_var = 'chi'
unit = '1/eV'

matplotlib.rc('xtick', labelsize = 14)
matplotlib.rc('ytick', labelsize = 14)

#Ploting of results
FigNum = 1    
if len(DifferentTerms) == 1:
    y_pred_rf = []

    if c == 'y':               
        y_pred_rf = rf_model.predict(Xpredict_fix)
        new_log = [np.log10(x) for x in y_pred_rf]

    else:
        y_pred_rf = rf_model.predict(Xpredict)
        #new_log = [np.log10(x) for x in y_pred_rf]
    
    if c == 'y':
        for value in range(len(y_pred_rf)):
            el_w.append(((1/(2*np.pi*2.9979))*1e-18*float(y_pred_rf[value])*(float(wavelen_fix[value]))**2)/10)
            print("Predicted w[nm] is: ({} nm,{} A)".format(el_w[value], wavelen_fix[value]))
        del el_w[0:]
    
    #Ploting of the results
            
    py.figure(figsize = (6.4,6.2))

    emitername = input("Enter name of emiter in form(e.g. Na I): ")

    #Geting Z and Ne
    si = emitername.index(" ")
    Z = PSE.get(emitername[0:si])
    N = input("Enter the electron density: ")
    T = int(input("Temperature: "))

    #Selecting available data from database
    dataI = data.loc[data['Emiter']==Z]
    dataII = dataI.loc[dataI['Ne'] == float(N)]
    dataIII = dataII.loc[dataII['Temp'] == T]
    
    #Loading data 
    ElectronW = dataIII['w'].values.tolist()
    chi = dataIII['chi'].values.tolist()
    wrad = dataIII['wrad'].values.tolist()

    #Defining paths for transition ploting
    pathNIST = r"C:\Users\ivant\Desktop\Python\NIST integration\ProbaNISTformerging"
    pathStark = r"C:\Users\ivant\Desktop\Python\NIST integration\StarkProba"

    #Creating MyTransition object and geting all transitions user wants to plot
    MyTransition = TransitionFinding(pathNIST, pathStark)
    transitions, TransitionTerm = MyTransition.FindTransitions(emitername, str(N), T)

        
    #Prompting user for transitions
    prompt1 = input("Number of wanted transitions to search (or if you want all, just type All): ")
    wantedTransition = []
    if prompt1 != "All":
        for i in range(int(prompt1)):
            t = input("Transition {}: ".format(i))
            wantedTransition.append(t)
    
        #Geting all wanted transitions 
        all_transitions, wantedResult = MyTransition.MatchingTransitions(transitions, wantedTransition, TransitionTerm)
        all_transitions = MyTransition.RemoveNans(all_transitions)

    else:
        wt = []
        for j in range(len(transitions)):
            lineloc = transitions[j].index("-")
            secondPart = transitions[j][lineloc+1:]
            
            if len(secondPart) == 2:
                s = secondPart.replace(secondPart[0],"n")
                wt.append(transitions[j][0:lineloc+1]+s)
            
            elif len(secondPart) == 3:
                s = secondPart.replace(secondPart[0:2],"n")
                wt.append(transitions[j][0:lineloc+1]+s)
        
        for tran in wt:
            if tran not in wantedTransition:
                wantedTransition.append(tran)

        #Geting all wanted transitions
        all_transitions, wantedResult = MyTransition.MatchingTransitions(transitions, wantedTransition, TransitionTerm)
        all_transitions = MyTransition.RemoveNans(all_transitions)

    #First get all distinct states
    marker_list = ["ks", "bo", "r^", "gv", "c<", "y>", "m*", "D", "d", "."] #Marker sings for ploting
    wTerm = []
                
    for vector in all_transitions:
        if vector[0][0]  not in wTerm:
            wTerm.append(vector[0][0])

    states_list = [[] for k in range(len(wTerm))] #For every wanted transitions one get a tuple (trans, wrad, chi)

    for j in range(len(wTerm)):
        for vectors in all_transitions:
            if wTerm[j] == vectors[0][0]:
                indexW = ElectronW.index(vectors[1])
                states_list[j].append((vectors[0][1],wrad[indexW], chi[indexW]))
                        
    chi_list = [[] for j in range(len(states_list))]
    wrad_list = [[] for j in range(len(states_list))]
    TNames = [[] for j in range(len(states_list))]

    counter = 0 #Keep track of marker index
    empty = 0 #Keep track so every simbol on graph is diferent

    for h in range(len(wTerm)):
        
        List = states_list[h]

        trans = wTerm[h]
                    
        for vector in List:
            chi_list[h].append(vector[2])
            wrad_list[h].append(vector[1])
            TNames[h].append(vector[0])

        if counter > len(marker_list)-1:
            counter = 0
            empty = 1

        py.loglog(chi_list[h], wrad_list[h], '{}'.format(marker_list[counter]), markersize = 6.5, label = '{}'.format(trans))
        py.legend(loc = 'lower right',fontsize = 10)

        for j in range(len(chi_list[h])):
            py.text(chi_list[h][j]-0.04, wrad_list[h][j]*3, "{}".format(TNames[h][j]), fontsize = 12, rotation = 90)
                            
        counter += 1      

    mult = multiplet.get(str(DifferentTerms[0]))
        
    #If transitions are fixed 
    if c == 'y':
        py.plot(log_chi,new_log,'ko',markersize = 6.5, markerfacecolor = 'none', markeredgecolor = 'black', label = 'Predicted data {} states'.format(mult))

        if OldData == True:
            py.plot(old_log_chi,old_log,'kv',markersize = 6.5, label = 'Old data {} states'.format(mult))
            for j in range(len(transition_fix)):
                if j == 0:
                    py.text(log_chi[j]-0.01, new_log[j]+0.12, "{}".format(transition_fix[j]), fontsize = 10, rotation = 90)
                    continue
                        
                if old_log[j-1]>new_log[j]:
                    py.text(log_chi[j]-0.01, new_log[j]+0.12, "{}".format(transition_fix[j]), fontsize = 10, rotation = 90)
                else:
                    py.text(log_chi[j]-0.01, old_log[j]+0.12, "{}".format(transition_fix[j]), fontsize = 10, rotation = 90)
                
        else:
            for j in range(len(transition_fix)):
                py.text(log_chi[j]-0.01, new_log[j]+0.12, "{}".format(transition_fix[j]), fontsize = 12, rotation = 90)
        
    #If they are not fixed 
    else:
        py.loglog(chi_new,y_pred_rf,'ko',markersize = 6.5, markerfacecolor = 'none', markeredgecolor = 'black', label = 'Predicted data {} states'.format(mult))
        if OldData == True:
            py.plot(old_log_chi,old_log,'kv',markersize = 6.5, label = 'Old data {} states'.format(mult))
            for j in range(len(transition_fix)):
                if j == 0:
                    py.text(log_chi[j]-0.01, new_log[j]+0.12, "{}".format(TransitionName[j]), fontsize = 10, rotation = 90)
                    continue
                        
                if old_log[j-1]>new_log[j]:
                    py.text(log_chi[j]-0.01, new_log[j]-1, "{}".format(TransitionName[j]), fontsize = 10, rotation = 90)
                        
                else:
                    py.text(log_chi[j]-0.01, old_log[j]-1, "{}".format(TransitionName[j]), fontsize = 10, rotation = 90)

        else:
            for j in range(len(TransitionName)):
                py.text(chi_new[j]-0.03, y_pred_rf[j]*3.6, "{}({})".format(TransitionName[j], DifferentTerms[0]), fontsize = 12, rotation = 90)
        
    py.xlabel(r"$\chi^{-1}$"+" [{}]".format(unit), fontsize = 14)
    py.ylabel(r"$\omega$ [rad/s]", fontsize = 14)
    py.xlim((0.8,12))
    py.ylim((2e11, 5.1*max(y_pred_rf)))
    py.legend(loc='lower right',fontsize = 12)
    FigNum += 1
    py.savefig(r'C:\Users\ivant\Desktop\Python\MLFakultet\Li I Serije\{}{}pred{}.png'.format(emitername,modelname,series),bbox_inches='tight')
    py.show()

#-------------------------------------------------------------------- If there are more multiplets --------------------------------------------------------------------
else:
    y_pred_tree = [[] for j in range(len(DiferentTerms))]
    new_log = [[] for j in range(len(DiferentTerms))]
    if c == 'y':
        for j in range(len(DiferentTerms)):
            y_pred_tree[j] = tree_model.predict(Xpredict_fix[j])
            new_log[j] = [np.log10(x) for x in y_pred_tree[j]]
            for value in range(len(y_pred_tree[j])):
                el_w.append(((1/(2*np.pi*2.9979))*1e-18*float(y_pred_tree[j][value])*(float(wavelen_fix[j][value]))**2)/10)
                print("Predicted w[nm] is: ({} nm,{} A)".format(el_w[value], wavelen_fix[j][value]))
            del el_w[0:]

    else:
        for j in range(len(DiferentTerms)):
            y_pred_tree[j] = tree_model.predict(Xpredict[j])
            new_log[j] = [np.log10(x) for x in y_pred_tree[j]]
            for value in range(len(y_pred_tree[j])):
                el_w.append(((1/(2*np.pi*2.9979))*1e-18*float(y_pred_tree[j][value])*(float(WavelenList[j][value]))**2)/10)
                print("Predicted w[nm] is: ({} nm,{} A)".format(el_w[value], WavelenList[j][value]))
            del el_w[0:]

    #Ploting of the results
    figsize = input("Figure size: width, height: ")
    figsize = figsize.split(',')
    figsize = [int(element.strip()) for element in figsize]
    figsize = tuple(figsize)
            
    py.figure(figsize = figsize)
    print("Drawing graphical predictions for best model")
    emitername = input("Enter name of emiter in form(e.g. Na I): ")

    #Geting Z and Ne
    si = emitername.index(" ")
    Z = PSE.get(emitername[0:si])
    N = input("Enter the electron density: ")
    T = int(input("Temperature: "))

    #Selecting data from base
    dataI = data.loc[data['Emiter']==Z]
    dataII = dataI.loc[dataI['Ne'] == float(N)]
    dataIII = dataII.loc[dataII['Temp'] == T]
    
    #Loading data 
    ElectronW = dataIII['w'].values.tolist()
    chi = dataIII['chi'].values.tolist()
    wrad = dataIII['wrad'].values.tolist()

    #Defining paths for transition ploting
    pathNIST = r"/home/petar/Desktop/IvanML/ProbaNISTformerging"
    pathStark = r"/home/petar/Desktop/IvanML/StarkProba"
    #Creating MyTransition object and geting all transitions user wants to plot
    MyTransition = TransitionFinding(pathNIST, pathStark)
    transitions, TransitionTerm = MyTransition.FindTransitions(emitername, str(N), T)

        
    #Prompting user for transitions
    prompt1 = input("Number of wanted transitions to search (or if you want all, just type All): ")
    wantedTransition = []
            
    if prompt1 != "All":
        for i in range(int(prompt1)):
            t = input("Transition {}: ".format(i))
            wantedTransition.append(t)
    
        #Geting all wanted transitions 
        all_transitions, wantedResult = MyTransition.MatchingTransitions(transitions, wantedTransition, TransitionTerm)
        all_transitions = MyTransition.RemoveNans(all_transitions)

    else:
        wt = []
        for j in range(len(transitions)):
            lineloc = transitions[j].index("-")
            secondPart = transitions[j][lineloc+1:]
            if len(secondPart) == 2:
                s = secondPart.replace(secondPart[0],"n")
                wt.append(transitions[j][0:lineloc+1]+s)
            
            elif len(secondPart) == 3:
                s = secondPart.replace(secondPart[0:2],"n")
                wt.append(transitions[j][0:lineloc+1]+s)
        
        for tran in wt:
            if tran not in wantedTransition:
                wantedTransition.append(tran)

        #Getting all transitions
        all_transitions, wantedResult = MyTransition.MatchingTransitions(transitions, wantedTransition, TransitionTerm)
        all_transitions = MyTransition.RemoveNans(all_transitions)

    #First get all distinct states
    marker_list = ["ks", "bo", "r^", "gv", "c<", "y>", "m*", "D", "d", "."] #Marker sings for ploting
    wTerm = []
                
    for vector in all_transitions:
        if vector[0][0]  not in wTerm:
            wTerm.append(vector[0][0])

    states_list = [[] for k in range(len(wTerm))] #For every wanted transitions one get a tuple (trans, wrad, chi)

    for j in range(len(wTerm)):
        for vectors in all_transitions:
            if wTerm[j] == vectors[0][0]:
                indexW = ElectronW.index(vectors[1])
                states_list[j].append((vectors[0][1],wrad[indexW], chi[indexW]))
                        
    chi_list = [[] for j in range(len(states_list))]
    wrad_list = [[] for j in range(len(states_list))]
    TNames = [[] for j in range(len(states_list))]

    counter = 0 #Keep track of marker index
    empty = 0 #Keep track so every simbol on graph is diferent

    sign_list = []
    for m in range(len(wTerm)):       
        if m == 0:
            sign_list.append("minus")
            continue
                
        if sign_list[m-1] == "minus":
            sign_list.append("plus")
                
        else:
            sign_list.append("minus")

    for h in range(len(wTerm)):        
        List = states_list[h]
        trans = wTerm[h]
                    
        for vector in List:
            chi_list[h].append(np.log10(vector[2]))
            wrad_list[h].append(np.log10(vector[1]))
            TNames[h].append(vector[0])
                    
        coeficitens1  = np.polyfit(chi_list[h], wrad_list[h],1)
        polynomial1 = np.poly1d(coeficitens1)
        y_p = polynomial1(chi_list[h])

        if counter > len(marker_list)-1:
            counter = 0
            empty = 1

        py.plot(chi_list[h], wrad_list[h], '{}'.format(marker_list[counter]), markersize = 7.5, label = '{}'.format(trans))
        py.plot(chi_list[h], y_p, 'k--', linewidth = 1.5)
        py.legend(loc = 'lower right',fontsize = 14)

        for j in range(len(chi_list[h])):                  
            if sign_list[h] == "minus":
                py.text(chi_list[h][j]-0.01, wrad_list[h][j]-1, "{}".format(TNames[h][j]), fontsize = 12, rotation = 90)
            
            else:
                py.text(chi_list[h][j]-0.01, wrad_list[h][j]+0.18, "{}".format(TNames[h][j]), fontsize = 12, rotation = 90)

        counter += 1


    if c == 'y':
        sign_list = []
        for m in range(len(DiferentTerms)):
            if m == 0:
                sign_list.append("minus")
                continue
            
            if sign_list[m-1] == "minus":
                sign_list.append("plus")
            else:
                sign_list.append("minus")
                
        for j in range(len(DiferentTerms)):
            mult = multiplet.get(str(DiferentTerms[j]))
            py.plot(log_chi[j],new_log[j],'{}'.format(marker_list[j+4]),markersize = 8.5, markerfacecolor = 'none', markeredgecolor = 'black', label = 'Predicted data {} states'.format(mult))

            if OldData == True:
                py.plot(old_log_chi[j],old_log[j],'{}'.format(marker_list[j+2]),markersize = 8.5, label = 'Old data {} states'.format(mult))
                    
                #Writing terms on graph
                for k in range(len(transition_fix[j])):
                    if k == 0:
                        if sign_list[j] == "minus":
                            py.text(log_chi[j][k]-0.01, new_log[j][k]-1.05, "{}".format(transition_fix[j][k]), fontsize = 12, rotation = 90)
                            continue
                        else:
                            py.text(log_chi[j][k]-0.01, new_log[j][k]+0.18, "{}".format(transition_fix[j][k]), fontsize = 12, rotation = 90)
                            continue                                    

                    if old_log[j][k-1]>new_log[j][k]:                          
                        if sign_list[j] == "minus":
                            py.text(log_chi[j][k]-0.01, new_log[j][k]-1.05, "{}".format(transition_fix[j][k]), fontsize = 12, rotation = 90)
                        else:
                            py.text(log_chi[j][k]-0.01, old_log[j][k-1]+0.18, "{}".format(transition_fix[j][k]), fontsize = 12, rotation = 90)
                            
                    else:                           
                        if sign_list[j] == "minus":
                            py.text(log_chi[j][k]-0.01, old_log[j][k-1]-1.05, "{}".format(transition_fix[j][k]), fontsize = 12, rotation = 90)
                        else:
                            py.text(log_chi[j][k]-0.01, new_log[j][k]+0.18, "{}".format(transition_fix[j][k]), fontsize = 12, rotation = 90)
            
            else:
                for k in range(len(transition_fix[j])):
                    if sign_list[j] == "minus":
                        py.text(log_chi[j][k]-0.01, new_log[j][k]-1.05, "{}".format(transition_fix[j][k]), fontsize = 12, rotation = 90)
                    else:
                        py.text(log_chi[j][k]-0.01, new_log[j][k]+0.18, "{}".format(transition_fix[j][k]), fontsize = 12, rotation = 90)

    else:
        sign_list = []
        for m in range(len(DiferentTerms)):
            if m == 0:
                sign_list.append("minus")
                continue
            
            if sign_list[m-1] == "minus":
                sign_list.append("plus")
            else:
                sign_list.append("minus")

        for j in range(len(DiferentTerms)):
            mult = multiplet.get(str(DiferentTerms[j]))
            py.plot(log_chi[j],new_log[j],'ko',markersize = 8.5, markerfacecolor = 'none', markeredgecolor = 'black', label = 'Predicted data {} states'.format(mult))
                
            if OldData == True:
                py.plot(old_log_chi[j],old_log[j],'{}'.format(marker_list[j+2]),markersize = 8.5, label = 'Old data {} states'.format(mult))
                for k in range(len(transition_fix[j])):
                    if k == 0:
                        if sign_list[j] == "minus":
                            py.text(log_chi[j][k]-0.01, new_log[j][k]-1.05, "{}".format(TransitionList[j][k]), fontsize = 12, rotation = 90)
                            continue
                        else:
                            py.text(log_chi[j][k]-0.01, new_log[j][k]+0.18, "{}".format(TransitionList[j][k]), fontsize = 12, rotation = 90)
                            continue                                    

                    if old_log[j][k-1]>new_log[j][k]:
                        if sign_list[j] == "minus":
                            py.text(log_chi[j][k]-0.01, new_log[j][k]-1.05, "{}".format(TransitionList[j][k]), fontsize = 12, rotation = 90)
                        else:
                            py.text(log_chi[j][k]-0.01, old_log[j][k-1]+0.18, "{}".format(TransitionList[j][k]), fontsize = 12, rotation = 90)
                        
                    else:
                        if sign_list[j] == "minus":
                            py.text(log_chi[j][k]-0.01, old_log[j][k-1]-1.05, "{}".format(TransitionList[j][k]), fontsize = 12, rotation = 90)
                        else:
                            py.text(log_chi[j][k]-0.01, new_log[j][k]+0.18, "{}".format(TransitionList[j][k]), fontsize = 12, rotation = 90)
            
            else:
                for k in range(len(transition_fix[j])):
                    if sign_list[j] == "minus":
                        py.text(log_chi[j][k]-0.01, new_log[j][k]-1.05, "{}".format(TransitionList[j][k]), fontsize = 12, rotation = 90)
                    else:
                        py.text(log_chi[j][k]-0.01, new_log[j][k]+0.18, "{}".format(TransitionList[j][k]), fontsize = 12, rotation = 90)
        
    py.xlabel(r"$\log(\{}$".format(indp_var)+r"$^{-1}$"+" [{}])".format(unit), fontsize = 16)
    py.ylabel(r"$\log(\omega$ [rad/s])", fontsize = 16)
    py.ylim((9,16))
    py.title("Fig {}. {}, Temp = {} K, Tree Regression method".format(FigNum,emitername,temp),fontsize = 16)
    py.legend(loc='best',fontsize = 11)
    py.savefig('C:/Users/ivant/Desktop/Python/MLFakultet/{}{}pred.png'.format(emitername,modelname), bbox_inches = 'tight')
    py.show()
    
#######################################################################################################################################
###################################################     Plotting the predictions      #################################################
###################################################  of the model using interpolation #################################################
###################################################     between input parameters      #################################################
#######################################################################################################################################

#######################################################################################################################################
##############################################################  Data needed ###########################################################
#######################################################################################################################################

emiter = 'Li'
emitername = 'Li I'
T = 30000
N = '1e16'

Z = PSE.get(emiter)
matplotlib.rc('xtick', labelsize = 14)
matplotlib.rc('ytick', labelsize = 14)

#Filtriranje podataka iz baze
dataI = data.loc[data['Emiter']==Z]
dataII = dataI.loc[dataI['Ne'] == float(N)]
dataIII = dataII.loc[dataII['Temp'] == T]

#Ucitavanje podataka
ElectronW = dataIII['w'].values.tolist()
chi = dataIII['chi'].values.tolist()
wrad = dataIII['wrad'].values.tolist()

###################################################################
######################  Transition finding ########################
###################################################################

#Paths for transitions (to the folders)
pathNIST = r"C:/Users/ivant/Desktop/Python/NIST integration/ProbaNISTformerging"
pathStark = r"C:/Users/ivant/Desktop/Python/NIST integration/StarkProba"

#Instance of Transition finding class
MyTransition = TransitionFinding(pathNIST, pathStark)
transitions, TransitionTerms = MyTransition.FindTransitions(emitername, str(N), T)

#Selection of spectral series
prompt1 = input("Number of wanted transitions to search (or if you want all, just type All): ")
wantedTransition = []

if prompt1 != "All":
    for i in range(int(prompt1)):
        t = input("Transition {}: ".format(i))
        wantedTransition.append(t)

    all_transitions, wantedResult = MyTransition.MatchingTransitions(transitions, wantedTransition, TransitionTerms)
    all_transitions = MyTransition.RemoveNans(all_transitions)

else:
    wt = []
    for j in range(len(transitions)):
        lineloc = transitions[j][0].index("-")
        secondPart = transitions[j][0][lineloc+1:]
        if len(secondPart) == 2:
            s = secondPart.replace(secondPart[0],"n")
            wt.append(transitions[j][0][0:lineloc+1]+s)

        elif len(secondPart) == 3:
            s = secondPart.replace(secondPart[0:2],"n")
            wt.append(transitions[j][0][0:lineloc+1]+s)

    for tran in wt:
        if tran not in wantedTransition:
            wantedTransition.append(tran)

    all_transitions, wantedResult = MyTransition.MatchingTransitions(transitions, wantedTransition, TransitionTerms)
    all_transitions = MyTransition.RemoveNans(all_transitions)

###################################################################
######################  Preparation for  ##########################
######################   Interpolation   ##########################
###################################################################

#series choosing
spectral_series = []
for vector in all_transitions:
    if vector[0][0] not in spectral_series:
        spectral_series.append(vector[0][0])

dfs = [[] for j in range(len(spectral_series))]
elws = [[] for j in range(len(spectral_series))]

#Taking of Stark widths for every series
for j in range(len(spectral_series)):
    for vector in all_transitions:
        if spectral_series[j] == vector[0][0]:
            d = dataIII.loc[dataIII['w'] == vector[1]]
            elws[j].append(d['wrad'].values[0])

#Featues extraction
for j in range(len(elws)):
    my_data = dataIII.loc[dataIII['wrad'].isin(elws[j])]
    dfs[j].append(my_data)

###################################################################
########################   Tables for   ###########################
########################  Interpolation ###########################
###################################################################

new_dfs = [[] for j in range(len(dfs))] #Novi dataframeovi koji sadrze tacke izmedju ulaznih podataka
help_list = [] #Pomocna lista za kreiranje kolona
y_fixed = [[] for j in range(len(dfs))]

for j in range(len(dfs)):
    y_fixed[j] = dfs[j][0].wrad
    dfs[j][0] = dfs[j][0].drop(['TD', 'w', 'wrad'], axis=1)
    
    for column in dfs[j][0].columns:
        if column == "Emiter" or column == "Z" or column == "Temp" or column == "Ne" or column == "J_donje" or column == "J_gornje" or column == "nf" or column == "lf" or column == "li" or column == 'Osn nivo':
            help_list.append((column, np.linspace(min(dfs[j][0][column]), max(dfs[j][0][column]),100)))
        else:
            help_list.append((column, np.linspace(min(dfs[j][0][column]), max(dfs[j][0][column]+5),100)))
            
    newdf = pd.DataFrame()
    for k in range(len(help_list)):
        newdf[f'{help_list[k][0]}'] = help_list[k][1]
    
    new_dfs[j].append(newdf)
    
###################################################################
############################## Plotting ###########################
###################################################################

y_pred = [[] for j in range(len(new_dfs))]

for j in range(len(y_pred)):
    y_pred[j] = rf_model.predict(new_dfs[j][0])

saving_path = r'C:/Users/ivant/Desktop/Python/MLFakultet/Li I Serije'
for j in range(len(y_pred)):
    chi_fixed = dfs[j][0].chi
    chi_new = new_dfs[j][0].chi
    py.figure(figsize = (6.4,6.2))
    py.loglog(chi_fixed, y_fixed[j], 'ks', markersize = 5.0, label = str(spectral_series[j]))
    py.loglog(chi_new, y_pred[j], 'r-', markersize = 3.5, label = str(spectral_series[j])+' model prediction')
    py.xlabel(r'$\chi^{-1}$ [1/eV]', fontsize = 14)
    py.ylabel(r'$\omega$ [rad/s]', fontsize = 14)
    py.xlim((min(chi_fixed)-0.1, 10))
    py.ylim((min(y_fixed[j])*2/5,3e13))
    py.legend(loc='lower right', fontsize = 12)
    py.savefig(f'{saving_path}/{spectral_series[j]}.png', bbox_inches = 'tight')
    py.show()
