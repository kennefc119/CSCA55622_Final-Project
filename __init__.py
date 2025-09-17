import numpy as np
import scipy as sp
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import difflib
from sklearn.preprocessing import LabelEncoder

# Prepare the checking list & define mappings##########################################################################
def prepare_successful_bids_list():
    accepted = ["awarded", "ok", "win", "done"]
    return accepted

def prepare_string_checking_list():
    columns_to_check = [
    "Location",
    "ProjectScope",
    "Client",
    "ClientSizeCategory",
    "LeadSource",
    "TenderType",
    "BidTeamLead",
    ]

    valid_locations = ["Central & Western", "Wan Chai","Eastern","Southern","Yau Tsim Mong","Kowloon City","Wong Tai Sin","Sham Shui Po","Kowloon Bay (Kwun Tong)","Sai Kung","Sha Tin","Tai Po","North","Tuen Mun","Yuen Long","Tsuen Wan","Kwai Tsing (Kwai Chung)","Islands"]
    valid_project_scope = ["Residential building","Hospital","Shopping mall","Carpark","Power station","Bridge","Other civil","Commercial"]
    valid_clients = [f"Client{chr(ord('A') + i)}" for i in range(15)]
    valid_client_size_cat = ["SME", "Large", "Small"]
    valid_lead_sources = ["existing client","referral","public tender"] 
    valid_tender_type = ["RFP", "RFQ", "PIN", "Direct award", "Framework"]
    valid_bid_team_lead = [f"Lead{chr(ord('A') + i)}" for i in range(10)]
    arr_valid_values = [valid_locations, valid_project_scope, valid_clients, valid_client_size_cat, valid_lead_sources, valid_tender_type, valid_bid_team_lead]
    
    column_valid_values_map = {
    "Location": valid_locations,
    "ProjectScope": valid_project_scope,
    "Client": valid_clients,
    "ClientSizeCategory": valid_client_size_cat,
    "LeadSource": valid_lead_sources,
    "TenderType": valid_tender_type,
    "BidTeamLead": valid_bid_team_lead
    }


    return columns_to_check,arr_valid_values,column_valid_values_map

def prepare_numerical_columns():
    numerical_columns = ['BiddingPrice', 
                     'TotalGFA',
                     'AreaPerFloor',
                     'NumberOfStoreys',
                     'ComplexityIndex',
                     'ClientSizeNumeric',
                     'TeamTotal',
                     'TeamJuniors',
                     'TeamSeniors',
                     'TeamAssociates',
                     'TeamAvgExperienceYears',
                     'TenderIssuedMonth',
                     'TenderIssuedYear',
                     'TrueNrOfCompetitors',
                     ]
    return numerical_columns

def prepare_categorical_columns():
    categorical_columns = [
        "Location",
        "ProjectScope",
        "Client",
        "ClientSizeCategory",
        "LeadSource",
        "TenderType",
        "BidTeamLead",
        'IsGovernmentProject'
    ]
    return categorical_columns

def prepare_columns_to_drop():
    drop_columns = ["ProjectID", "Outcome", "TenderRegion", "TenderIssuedDate"]
    return drop_columns

def identify_imbanlance_class_to_remove():
    dict_imbalnace = {
        "ProjectScope": ["Other civil","Bridge","Power station","Hospital"],
        }
    return dict_imbalnace

def identify_imbanlance_class_to_undersample():
    dict_imbalnace = {
        "ClientSizeCategory": [["Large",0.7]],
        "TenderType": [["RFP",0.4]],
        "ProjectScope": [["Residential building",0.55]],
        "Client": [["ClientB",0.7],["ClientL",0.7]],
        }
    return dict_imbalnace

def identify_imbanlance_class_to_oversample():
    dict_imbalnace = {
        "ClientSizeCategory": [["SME",5],["Small",2]],
        "LeadSource": [["referral",2]],
        "ProjectScope": [["Carpark",2],["Shopping mall",2]],
        "Client": [["ClientM",2],["ClientE",2],["ClientF",2],["ClientG",2],["ClientO",2],["ClientD",2],["ClientC",2],["ClientJ",2]],
        "TenderType": [["PIN",2],["Direct award",2],["Framework",2]],
        }
    return dict_imbalnace

#For filtering for successful bids###########################################################################
def filter_successful_bids(df,string_list = prepare_successful_bids_list()):
    # unique_values = set(df["Outcome"])
    # print (f'"Outcome" columns contain {unique_values}')
    df = df[df["Outcome"].str.lower().isin(string_list)].copy()
    print (f'Shape of df after filtering for successful bids: {df.shape}')
    return df


#For String Columns Checking and normalization###########################################################################
def normalize_unique_values(df, check_list=prepare_string_checking_list()):
    columns_to_check, arr_valid_values, string_check_mapping = check_list

    for i, col in enumerate(columns_to_check):
        unique_values = set(df[col])
        print(f'"{col}" unique nr in dataset = {len(unique_values)} vs expected nr = {len(arr_valid_values[i])}')
        if len(unique_values) != len(arr_valid_values[i]):
            valid_list = string_check_mapping[col]
            df[col] = df[col].apply(lambda x: normalize_column(x, valid_list))

    return df

def normalize_column(name,valid_list):
    name = str(name).strip().lower()
    match = difflib.get_close_matches(name, [d.lower() for d in valid_list], n=1, cutoff=0.6)
    if match:
        return next(d for d in valid_list if d.lower() == match[0])
    return None


#For Numeric conversion###########################################################################
def to_numeric(df, columns = prepare_numerical_columns()):
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


#For Label Encoding###########################################################################
def encoding_categorical_columns(df,columns = prepare_categorical_columns()):
    label_encoders = {}
    for col in columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    return df, label_encoders


#For Column dropping and NaN removal###########################################################################
def drop_na_and_columns(df, drop_columns = prepare_columns_to_drop()):
    for col in drop_columns:
        if col in df.columns:
            df = df.drop(columns=[col])
    return df


#For Imbalance checking and Oversampling###########################################################################
def count_plot(df,column_name,filename = 'Count Plot.png'):
    plt.clf()
    sns.countplot(data=df, x= column_name)
    plt.xticks(rotation=90)
    plt.title(f'Class Distribution of {filename.split(".png")[0]}')
    plt.tight_layout()
    plt.savefig('img/' + filename, facecolor='white', bbox_inches='tight')
    # plt.show()

#For Imbalance adjustment###########################################################################
def adjust_imbalance(df):
    # Remove classes
    remove_dict = identify_imbanlance_class_to_remove()
    for col, classes_to_remove in remove_dict.items():
        df = df[~df[col].isin(classes_to_remove)]

    # Undersample classes
    undersample_dict = identify_imbanlance_class_to_undersample()
    for col, class_list in undersample_dict.items():
        for class_name, fraction in class_list:
            df_class = df[df[col] == class_name]
            df = df[df[col] != class_name]
            n_sample = int(len(df_class) * fraction)
            df = pd.concat([df, df_class.sample(n=n_sample, random_state=42)], ignore_index=True)

    # Oversample classes
    oversample_dict = identify_imbanlance_class_to_oversample()
    for col, class_list in oversample_dict.items():
        for class_name, factor in class_list:
            df_class = df[df[col] == class_name]
            if len(df_class) == 0:
                continue
            df_oversample = pd.concat([df_class] * (factor - 1), ignore_index=True)
            df = pd.concat([df, df_oversample], ignore_index=True)

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


#Functions for data visualization###########################################################################
def pairplot(df,filename="Pairplot.png"):
    plt.clf()
    graph = sns.pairplot(df, 
                diag_kind='kde',
                corner=False,
                kind='scatter',
                plot_kws={'s': 40},
                )
    for ax in graph.axes.flatten():
        if ax:
            ax.set_xticks([])
            ax.set_yticks([])
    for i, ax in enumerate(graph.axes[-1, :]):
        if ax:
            ax.set_xlabel(ax.get_xlabel(), fontsize=19)

    for i, ax in enumerate(graph.axes[:, 0]):
        if ax:
            ax.set_ylabel(ax.get_ylabel(), fontsize=19)

    graph.savefig('img/' +filename, facecolor='white', bbox_inches='tight')

def heatmap_correlation(df, filename="Correlation Matrix.png"):
    plt.clf()
    cor_df = df.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(cor_df, annot=True, fmt=".2f", cmap="Blues", square=True, cbar_kws={"shrink": 0.8})
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig('img/' +filename, facecolor='white', bbox_inches='tight')
    # plt.show()

def describe_data(df):
    print (f'number of projects: {df.shape[0]}')
    print (f'number of features: {df.shape[1]}')
    print(df.shape)
    print(df.info())
    df.describe(include='all')


#Main code starts here########################################################################
raw_df = pd.read_csv('data/synthetic_arup_bids_v20_full_shuffled.csv')

df = filter_successful_bids(raw_df)
df = normalize_unique_values(df)
df = to_numeric(df)

df = drop_na_and_columns(df)

print (set(df['IsGovernmentProject']))


col_names = ['ProjectScope','ClientSizeCategory','Location','Client','LeadSource','TenderType','BidTeamLead']
[count_plot(df, col_name,filename = 'Before adjustment' + col_name + '.png') for col_name in col_names]

df = adjust_imbalance(df)
[count_plot(df,col_name,filename = 'After adjustment' + col_name + '.png') for col_name in col_names]

df,_cat_encoder = encoding_categorical_columns(df)

#Describe data + Visualizations + Correlation heatmap
############################################################################################################
describe_data(df)
heatmap_correlation(df)
pairplot(df)
print ("END")



