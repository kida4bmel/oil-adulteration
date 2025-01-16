"""
Module for generating simulated oil data with montecarlo and saves into .csv format
"""

import numpy as np
import pandas as pd
import dataloading

class MonteCarlo_simulator:
    def __init__(self, oil1, oil2, label1, label2, columns):
        self.oil1 = oil1 #any kind of oil can be used
        self.oil2 = oil2
        self.label1 = label1
        self.label2 = label2
        self.columns = columns

    def get_data_montecarlo(self, oil_data):
        """
        Gets mean values and covariance matrices of the oil samples
        """
        oil_data_no_label = oil_data.drop("Label", axis=1)
        oil_data_values = oil_data_no_label.values 
        covariance_matrix = np.cov(oil_data_values,rowvar=False) 
        mean_values = oil_data.describe().iloc[1].to_numpy()
        return mean_values, covariance_matrix
    
    def montecarlo_simulation(self, data_mean, data_cov, num_simulations, *args):
        """
        Samping from multivariate distribution
        """
        np.random.seed(8) #to ensure reproducibility
        simulations= np.random.multivariate_normal(data_mean, data_cov, num_simulations)
        if args:
            simulations = pd.DataFrame(simulations, columns=self.columns[1:] )
            labelname= args[0] 
            simulations["Label"]  = labelname[0] 

        return simulations
    
    def generate_mixtures(self, oil1, oil2, ratio_split, labels):
        """
        Weighted sum for generating mixtures
        """
        all_mixtures =[] 
        for i in range(len(ratio_split)):
            oils =[] 
            oil1_ratio = np.multiply(oil1, ratio_split[i] [0] )
            oil2_ratio = np.multiply(oil2, ratio_split[i] [1] ) 
            oils_mixture = oil1_ratio + oil2_ratio
            oils.append(oils_mixture)
            oils_np = np.row_stack(oils)
            mixed_oils = pd.DataFrame(oils_np, columns=list(self.columns[1:]  ))
            mixed_oils["Label"]  = labels[i] # add label column
            all_mixtures.append(mixed_oils)
        mixed_oils_all = pd.concat(all_mixtures)

        return mixed_oils_all
 
    def thresholds(self, oil_dataframe, threshold1, threshold2):
        """
        Consider thresholds (eventually not needed in all cases)
        """
        for column in list(self.columns[1:] )[0:30]:
            oil_dataframe[column] = np.where(oil_dataframe[column] < threshold1, 0, oil_dataframe[column])
        for column in list(self.columns[1:] )[30:] :
            oil_dataframe[column] = np.where(oil_dataframe[column] < threshold2, 0, oil_dataframe[column])

        return oil_dataframe
    
    def create_final_dataset( self, pure_oil1_sims, pure_oil2_sims, mixtures):
        """
        Generates a pandas dataframe with pure and mixed simulations
        """
        dataset = pd.concat([pure_oil1_sims, pure_oil2_sims, mixtures] )
        
        return dataset
    
    def all_steps_together(self, num_simulations, ratios, labels, threshold1, threshold2):
        """
        Performs all steps in one to avoid calling all functions individually
        """
        mean_oil1, cov_oil1 = self.get_data_montecarlo(self.oil1)
        mean_oil2, cov_oil2 = self.get_data_montecarlo(self.oil2)

        sims_oil1 = self.montecarlo_simulation(mean_oil1, cov_oil1,num_simulations)
        sims_oil2 = self.montecarlo_simulation(mean_oil2, cov_oil2,num_simulations)
        mix_co_so = self.generate_mixtures(sims_oil1, sims_oil2, ratios, labels)

        pure_sims_oil1= self.montecarlo_simulation(mean_oil1,cov_oil1 , num_simulations, self.label1)
        pure_sims_oil2= self.montecarlo_simulation(mean_oil2,cov_oil2 , num_simulations, self.label2)

        mix_oils_threshold = self.thresholds(mix_co_so, threshold1, threshold2)
        pure_sims_oil1_threshold = self.thresholds(pure_sims_oil1, threshold1, threshold2)
        pure_sims_oil2_threshold = self.thresholds(pure_sims_oil2, threshold1, threshold2)

        data_classifier = self.create_final_dataset(pure_sims_oil1_threshold, pure_sims_oil2_threshold, mix_oils_threshold)

        return data_classifier
    
    def save_to_csv(self, dataset_pd, path):
        """
        saves data into .csv format
        """
        dataset_pd.to_csv(path)
        print("File saved")

        return None


if __name__ == "__main__":
    num_sim=100
    ratios = [[0.99, 0.01], [0.97, 0.03],[0.95, 0.05],[0.93, 0.07] ,[0.91, 0.09]   ] 
    thres1 = 0.05
    thres2 = 0.23
    path_to_csv = "oil_dataset.csv" # enter own path

    path_to_excel_co = 'CO.xlsx'
    path_to_excel_so = 'SO.xlsx'
    path_to_excel_realmix = "realmixtures.xlsx"
    loaded_data_co = dataloading.LoadOilData(path_to_excel_co)
    loaded_data_so = dataloading.LoadOilData(path_to_excel_so)
    loaded_data_real =  dataloading.LoadOilData(path_to_excel_realmix)
    co = loaded_data_co.get_dataframe()
    co_label = loaded_data_co.return_label()
    so =  loaded_data_so.get_dataframe()
    so_label = loaded_data_so.return_label()
    feats = loaded_data_co.return_features()
    real = loaded_data_real.get_dataframe()
    labelreal = loaded_data_real.return_label()

    data_for_mc = MonteCarlo_simulator(co, so, co_label, so_label, feats)

    # Option 1: do steps one after another
    # mean_co, cov_co = data_for_mc.get_data_montecarlo(co)
    # mean_so, cov_so = data_for_mc.get_data_montecarlo(so)
    # mc_co = data_for_mc.montecarlo_simulation(mean_co, cov_co, num_sim  )
    # mc_so = data_for_mc.montecarlo_simulation(mean_so, cov_so, num_sim)
    # co_pure = data_for_mc.montecarlo_simulation(mean_co, cov_co, num_sim, co_label)
    # so_pure = data_for_mc.montecarlo_simulation(mean_so, cov_so, num_sim, so_label)
    # mix = data_for_mc.generate_mixtures(mc_co, mc_so, ratios, labelreal)
    # mix_threshold = data_for_mc.thresholds(mix, 0.05, 0.23)
    # sims_co_threshold = data_for_mc.thresholds(co_pure, thres1, thres2)
    # sims_so_threshold = data_for_mc.thresholds(so_pure, thres1, thres2)
    # dataset_classification = data_for_mc.create_final_dataset(sims_co_threshold, sims_so_threshold, mix_threshold)

    # Option 2: doing all steps in one
    dataset_classification = data_for_mc.all_steps_together(num_sim, ratios, labelreal, thres1, thres2)
    
    data_for_mc.save_to_csv(dataset_classification, path=path_to_csv)
