def get_column_values_list(df, clm_name, indices_list):
  """
  Returns a list of values from dataframe `df` of a particular column `clm_name`.

  Args:
    df (pandas.DataFrame): A dataframe from which values would be extracted.
    clm_name (str): The column name whose values are required.
    indices_list (str): Row indices values in column `clm_name`.

  Returns:
    []: A list of required values of the column.
  """
  return df[clm_name].loc[[indices_list]].tolist()
