"""
Loads the data
Requirements:
Dataformat Excel
1 file per oil class
1 file with actual mixtures
Label column must be present in all files
Same amount of features for all oils
"""

import pandas as pd

class LoadOilData:
  def __init__(self, excel_file):
    self.excel_file = excel_file

  def get_dataframe (self):
    """
    Returns a dataframe from the Excel file with the oil data
    """
    df_oil = pd.read_excel(self.excel_file )

    return df_oil

  def return_features(self):
    """
    Returns feature names
    """
    df_oil = self.get_dataframe()
    columns = df_oil.columns

    return columns
 
  def return_label(self):
    """
    Returns a list with the label(s)
    """
    df_oil = self.get_dataframe()
    label = df_oil.iloc[:,0:1].to_numpy().flatten().tolist()
    labels= list(dict.fromkeys(label))

    return labels
  
if __name__ == "__main__":
    path_to_excel1= 'CO.xlsx'
    path_to_excel2= 'SO.xlsx'
    path_to_excel_real_mixtures= "realmixtures.xlsx"

    loaded_data = LoadOilData(path_to_excel_real_mixtures)
    pd_df = loaded_data.get_dataframe()
    features = loaded_data.return_features()
    label = loaded_data.return_label()
    
