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

def get_number_of_subjects_in_age_range_int(df, clm_name="", age_start=None,
    age_end=None):
  if clm_name=="":
    print "Column name is empty. Feed a non empty valid column name."
    return

  if age_start == None and age_end == None:
    print "Both age_start and age_end are None, invalid input."
    return

  if age_end == None:
    ans = sum(df[clm_name].dropna() >= age_start)
    print "Number of subjects with age greater than or equal to: %s is %s" % (
        age_start, ans)
    return

  if age_start == None:
    ans = sum(df[clm_name].dropna() <= age_end)
    print "Number of subjects with age lesser than or equal to: %s is %s" % (
        age_end, ans)
    return

  if age_start > age_end:
    print "Age start: %s is greater than age end: %s" % (age_start, age_end)
    return

  grtr_e_lst = (df[clm_name].dropna() >= age_start).tolist() # Bool list.
  lssr_e_lst = (df[clm_name].dropna() <= age_end).tolist() # Bool list.
  ans = sum([ x&y for (x,y) in zip(grtr_e_lst, lssr_e_lst)])
  print "Number of subjects in age range [%s, %s] is: %s" % (
      age_start, age_end, ans)
  return
