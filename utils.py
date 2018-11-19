import inspect

#calculo
import numpy as np
import pandas as pd

#grafico
import matplotlib.pyplot as plt
from IPython.display import display_html

set_parameter_csv = {
    'sep': ',',
    'encoding': 'ISO-8859-1',
    'low_memory': False
}

def get_memory_usage(data, deep=True):
    return '{} MB'.format(data.memory_usage(deep=deep).sum() / 1024 ** 2)

def null_verificator(data):        
    if data.isnull().any().any():
        view_info = pd.DataFrame(
            pd.concat(
                [data.isnull().any(), 
                 data.isnull().sum(),
                 data.dtypes], 
                axis=1)
        )
        view_info.columns = ['Nulos', 'Cantidad', 'Tipo Col']
        size = data.shape[0]
        view_info['Porcentaje'] = view_info['Cantidad'].apply(
            lambda x: str(np.round(0 if not x else x*100 / size, 2)) + ' %')
        return view_info
    else:
        return "DATA LIMPIA DE NULOS"

def reduce_size_data(df, category=False, default=''):
    print("Tamaño de uso actual : ", get_memory_usage(df))
    print("-> Int 64 Detected")
    for col in df.select_dtypes(include=['int']).columns:
        print(" "*4, col)
        df[col] = pd.to_numeric(arg=df[col], downcast=default or'integer')
    
    print("-> Float 64 Detected")
    for col in df.select_dtypes(include=['float']).columns:
        print(" "*4, col)
        df[col] = pd.to_numeric(arg=df[col], downcast=default or'float')
    
    if category:
        for col in df.select_dtypes(include=['object']).columns:
            if len(df[col].unique()) / len(df[col]) < 0.5:
                df[col] = df[col].astype('category')
                
    print("Tamaño de uso final : ", get_memory_usage(df))                
    return df


def  add_col_dates(data, col, format_match="%d-%b-%y", month=True, day=True, year=True,
                   weekday=True, replace_str=False, format_str_replace='%Y/%m/%d', replicate_in_test=False):
    """
        por optimizar en casos separados para data y test_data
    """
    data['date'] = pd.to_datetime(data[col], format=format_match)

    if month:
        data['month'] = pd.to_numeric(data['date'].dt.strftime('%m'), errors='coerce')
        data['month'].fillna(-99)  
    if day:
        data['day'] = pd.to_numeric(data['date'].dt.strftime('%d'), errors='coerce')
        data['day'].fillna(-99)
    if year:
        data['year'] = pd.to_numeric(data['date'].dt.strftime('%Y'), errors='coerce')
        data['year'].fillna(-99)
    if weekday:
        data['weekday'] = pd.to_numeric(data['date'].dt.strftime('%w'), errors='coerce')
        data['weekday'].fillna(-99)
    if replace_str:
        data['date'] = data['date'].dt.strftime(format_str_replace)
        
    return(data.head(10))

def analysis_for_group(data, list_init, col_output, cant_view=40):
    analysis_df =  pd.concat([
        data.groupby(by=list_init)[col_output].count(),
        data.groupby(by=list_init)[col_output].sum(),
        data.groupby(by=list_init)[col_output].mean(),
        data.groupby(by=list_init)[col_output].max(),
        data.groupby(by=list_init)[col_output].min(),
        data.groupby(by=list_init)[col_output].skew()
    ], axis=1)
    analysis_df.columns = ['Total','Suma', 'Media', 'Max', 'min', 'Sesgo']
    return analysis_df.head(cant_view)

def get_default_args(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }

def serie_ratio_convergencia(data, var, col_target='TARGET'):
    df = data.groupby(
        by=[var]
    ).mean()[col_target]
    del df.index.name
    return df

def replace_value_ratio(train, test, lista_vars, col_target='target'):
    for var in lista_vars:
        print('>> ', var.upper())
        serie_ratio = serie_ratio_convergencia(
            train, var, col_target=col_target
        )
        print(train[var].unique())
        print(test[var].unique())
        train[var + '_ratio'] = train[var].apply(lambda x: serie_ratio.get(x, 0))
        test[var + '_ratio'] = test[var].apply(lambda x: serie_ratio.get(x, 0))
    return train, test

def ratio_convergencia(data, var, col_id='ID_CLIENTE', col_target='TARGET'):
    print("--------------------------------------------------------")
    print("Var", var,":")
    df = data.groupby(
        by=[var], 
        as_index=False
    ).agg(
        {col_id:'count', col_target:'mean'}
    ).rename(
        columns= {col_id:'FREC', col_target: 'RT_CONVERSION'}
    )
    return df


def compare_categories(cols, train, test):
    detected = 0
    for col in cols_str:
        diff_cat = set(test[col].unique()) - set(train[col].unique())
        if diff_cat:
            detected += 1
            print(col.upper(), diff_cat)
            
    if detected == 0:
        print("CATEGORIAS HOMOGENEAS EN TRAIN Y TEST")


def display_horizontal(*args, percent_sep=5):
    html_str=''
    for table in args:
        df = table if isinstance(table, pd.DataFrame) else  pd.DataFrame(table)
        html_str+=df.to_html()
    display_html(
        html_str.replace(
            'table','table style="display:inline;padding-right:{}%"'.format(percent_sep)
        ), 
        raw=True)
