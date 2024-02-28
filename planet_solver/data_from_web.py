'''This module aims to get the data from the respective database, using https://mast.stsci.edu/api/v0/MastApiTutorial.html'''

import sys
import os
import time
import re
import json

import requests
from urllib.parse import quote as urlencode

from astropy.table import Table
import numpy as np
import pandas as pd
import pprint
pp = pprint.PrettyPrinter(indent=4)



def mast_query(request):
    """Perform a MAST query.
    
        Parameters
        ----------
        request (dictionary): The MAST request json object
        
        Returns head,content where head is the response HTTP headers, and content is the returned data"""
    
    # Base API url
    request_url='https://mast.stsci.edu/api/v0/invoke'    
    
    # Grab Python Version 
    version = ".".join(map(str, sys.version_info[:3]))

    # Create Http Header Variables
    headers = {"Content-type": "application/x-www-form-urlencoded",
               "Accept": "text/plain",
               "User-agent":"python-requests/"+version}

    # Encoding the request as a json string
    req_string = json.dumps(request)
    req_string = urlencode(req_string)
    
    # Perform the HTTP request
    resp = requests.post(request_url, data="request="+req_string, headers=headers)
    
    # Pull out the headers and response content
    head = resp.headers
    content = resp.content.decode('utf-8')

    return head, content


def set_filters(parameters):
    return [{"paramName":p, "values":v} for p,v in parameters.items()]


def set_min_max(min, max):
    return [{'min': min, 'max': max}]


object_of_interest = 'M101'

resolver_request = {'service': 'Mast.Name.Lookup',
                    'params': {'input': object_of_interest,
                               'format': 'json'},
                    }

headers, resolved_object_string = mast_query(resolver_request)

resolved_object = json.loads(resolved_object_string)

# pp.pprint(resolved_object)
obj_ra  = resolved_object['resolvedCoordinate'][0]['ra']
obj_dec = resolved_object['resolvedCoordinate'][0]['decl']


mast_request = {'service': 'Mast.Caom.Cone',
                'params':  {'ra': obj_ra,
                            'dec': obj_dec,
                            'radius': 0.2},
                'format':  'json',
                'pagesize': 2000,
                'page': 1,
                'removenullcolumns': True,
                'removecache': True}

headers, mast_data_str = mast_query(mast_request)

mast_data = json.loads(mast_data_str)

# print(mast_data.keys())
# print("Query status:", mast_data['status'])
# pp.pprint(mast_data['data'][0])




def put_query_in_astropy_table_format(mast_data):
    '''This function gets the query and returns the outcome in an Astropy-like table'''

    mast_data_table = Table()

    for col, atype in [(x['name'], x['type']) for x in mast_data['fields']]:
        if atype == "string":
            atype = "str"
        if atype == "boolean":
            atype = "bool"
        mast_data_table[col] = np.array([x.get(col, None) for x in mast_data['data']] , dtype = atype)

    return mast_data_table


def make_df_from_astropy_table(mast_data_table):
    '''This function turns an astropy table into a pandas dataframe'''

    df = mast_data_table.to_pandas()
    return df


def download_data_to_csv(file_name = "data.csv"):
    '''This function downloads data to a csv file into a pre-determined location'''

    directory = r"C:\\Planetary_Solver\\input\\"
    file_path = directory + file_name
    df.to_csv(file_path, index = False)



# Run the code
mast_data_table= put_query_in_astropy_table_format(mast_data)
df             = make_df_from_astropy_table(mast_data_table)
download_data_to_csv(file_name = "data.csv")
