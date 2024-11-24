# 2024, The DEPA CCR DP Non Network Model Implementation
# authors shyam@ispirt.in, sridhar.avs@ispirt.in
#
# Licensed TBD
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Key references / Attributions: https://github.com/YangLi069/DPEL

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import importlib.util
import sys
import os
import json

#loading the pipline config 
with open('/Users/sridharaluru/Desktop/depa_non_network_models/config/pipeline_config.json', 'r') as file:
    data = json.load(file)

# Define the full path to the file you want to import
tree_file_path = data["pipeline"][0]["config"]["saved_model_path"]
dataset_path=data["pipeline"][0]["config"]["input_dataset_path"]
target_variable=data["pipeline"][0]["config"]["target_variable"]
logger_file=data["pipeline"][0]["config"]["logger_file"]
test_size=data["pipeline"][0]["config"]["test_size"]
random_state=data["pipeline"][0]["config"]["random_state"]

def write_to_file(text,logger_file):
    # Open the file in write mode and write the text
    with open(logger_file, "a") as file:
        file.write(text)

def get_saved_model(tree_file_path,model_name):
    #we are loading the saved model 
    module_name = os.path.basename(tree_file_path).replace('.py', '')
    spec = importlib.util.spec_from_file_location(module_name, tree_file_path)
    tree_module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = tree_module
    spec.loader.exec_module(tree_module)
    return getattr(tree_module,model_name)


def fetch_dataset(dataset_path):
    df=pd.read_csv(dataset_path)
    print("Dataset Loaded")
    df=df.apply(pd.to_numeric,errors='coerce')
    df.dropna(inplace=True)
    return df



def train(dataframe,target_variable,test_size,random_state,model,logger_path):
    X = df.drop(target_variable, axis=1) 
    y = df[target_variable] 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    write_to_file("X_train shape: {}\n".format(X_train.shape), logger_file)
    write_to_file("X_test shape: {}\n".format(X_test.shape), logger_file)
    write_to_file("y_train shape: {}\n".format(y_train.shape), logger_file)
    write_to_file("y_test shape: {}\n".format(y_test.shape), logger_file)
    write_to_file("Initializing Decision Tree Model\n", logger_file)
    tree=model(gt_privacy_p=float(1.0/100))
    write_to_file("Tree Setup For the model\n",logger_file)
    tree.fit(X_train,y_train)
    write_to_file("Model Fitting\n",logger_file)
    pred1=tree.predict(X_test)
    write_to_file("Model Prediction Complete\n",logger_file)
    write_to_file(str(pred1),logger_file)

#actual run 
dt=get_saved_model(tree_file_path,'DecisionTree')
df=fetch_dataset(dataset_path)
train(df,target_variable,test_size,random_state,dt,logger_file)