# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 21:15:45 2019

@author: josh
"""
import json
import pandas as pd
import glob
import os
import numpy as np

df = pd.read_csv('anonymized_playtest6_data.csv', sep=';')

#%%
"""
Types of shape:
"""
shape_types = {1 : 'cube',
               2 : 'square_pyramid',
               3 : 'ramp',
               4 : 'cyclinder',
               5 : 'cone',
               6 : 'sphere'}

#%%
from typing import Tuple

class solutions_summary():
    
    def __init__(self, excluded_files : list=None) -> None:   
        if excluded_files is None:
            self.excluded_files = ['config.json', 'blank.json']
        else:
            self.excluded_files = excluded_files
            
        ## store a simplified view of the solution file data
        self.solutions_summary = {}
        self.solutions_shape_data = {}
        
        self.solutions_shape_type ={}
        self.solutions_coords = {}


    def _count_shapes(self, shape_data : dict) -> dict:
        """
        find the number of each shape in the puzzle
        """
        shape_count = {}
        for item in shape_data:
            item_shape_type = item.get('shapeType')
            if item_shape_type not in shape_count:
                shape_count[item_shape_type] = 1
            else:
                shape_count[item_shape_type] += 1
        return shape_count


    def _extract_solution_data(self, shape_data : list) -> Tuple[list,list]:
        """
        extract x,y,z position, info from: list of dicts, into numpy array for ease of use
        also insert the shapeType
        """
        
        pos_op = []
        shapetype_op = []
        
        for shape in shape_data:
            pos = shape.get('gridPosition')
            coord = np.array([pos['x'], 
                              pos['y'], 
                              pos['z']])
            
            
            
            pos_op.append(coord)
            shapetype_op.append(shape.get('shapeType'))
        return pos_op, shapetype_op
        

    def summarise(self) -> Tuple[dict,dict]:
        
        for solution_file in glob.glob('StreamingAssets/*.json'):
            
            ## split the path by the platform's path separator, take the filename only
            filename = solution_file.split(os.sep)[-1]
            ## avoid processing the excluded files
            if filename in self.excluded_files:
                continue
            
            
            solution_data = json.load( open(solution_file) )
            shape_data = solution_data.get('shapeData')
            unique_shape_types = list( pd.Series( [x.get('shapeType') for x in shape_data] ).unique() )
            shape_counts = self._count_shapes(shape_data=shape_data)
            
            ## store the results
            self.solutions_summary[solution_data.get('puzzleName')] = {
#                    'shape_data'         : shape_data,
                    'unique_shape_types' : unique_shape_types,
                    'shape_counts'       : shape_counts}
            
            extracted_shape_data = self._extract_solution_data(shape_data)
#            self.solutions_shape_data[solution_data.get('puzzleName')] = extracted_shape_data
            key = solution_data.get('puzzleName')
            self.solutions_shape_type[key]  = extracted_shape_data[1]
            self.solutions_coords[key]      = extracted_shape_data[0]
        
        return self.solutions_summary, self.solutions_coords, self.solutions_shape_type


     
#%%
summary,coords,shapes = solutions_summary().summarise()

#%%
"""
checks:
    number of shapes match
    number of each type of shape matches
    (mean) distance     from candidate nearest shape of the same type
    (mean) angle        difference from the nearest shape of the same type
    (mean) scale        difference from the nearest shape of the same type 
    
    
coordinate transforms to center zero as the average centroid of all of the shapes
"""
#%%


class coord_trans():
    
    def __init__(self,
                 shape_coords   : np.array, 
                 shape_types    : list,
                 puzzle_id      : str) -> None:
        self.shape_coords = shape_coords
        self.shape_types = shape_types
        self.shape_type_mapping = {
                1 : 'cube',
                2 : 'square_pyramid',
                3 : 'ramp',
                4 : 'cyclinder',
                5 : 'cone',
                6 : 'sphere'}
        self.puzzle_id = puzzle_id
        
        
        ## information on the known solution
        self.sol_summary,\
        self.sol_pos,\
        self.sol_shapetype = solutions_summary().summarise()



    def _find_shapetype_coords(self, 
                               shape_type : int) -> np.array:
        """
        Return the coords for all shapes of a given shapeType
        then we can find the nearest neighbours 
        """
        
        coord_indicies = []
        for index, value in enumerate(self.shape_types):
            if value == shape_type:
                coord_indicies.append(index)
        
        return_value = []
        for index in coord_indicies:
            return_value += [self.shape_coords[index]] 
        
        return np.array( return_value )
    
        
    def _find_coord_centre(self,
                           shape_coords) -> np.array:
        """
        find centre of coordinate system
        """
        return shape_coords.mean(axis=0)
    
        
    def _transform_coords(self,
                          shape_coords : np.array) -> np.array:
        """
        transform using the coordinate centre
        """
        
        coord_centre = self._find_coord_centre(shape_coords)
        return shape_coords - coord_centre
    
    
    def _euclidean_distance(self,
                            desired_pos : np.array,
                            actual_pos  : np.array) -> float:
        """
        return euclidean distance
        """
        return np.sqrt(
                        np.sum( 
                                (desired_pos-actual_pos)**2  
                               )
                        )
    
    def _vector_direction(self,
                          desired_pos : np.array,
                          actual_pos  : np.array) -> np.array:
        """
        find the direction they would need to move their shape to get to the right answer
        """
        return desired_pos - actual_pos


    def transform_all_coords(self,
                             shape_coords : np.array) -> np.array:
        """
        public fn: transform all coords
        """
        return self._transform_coords(shape_coords)
    
    
    def transform_all_shapetype(self, 
                                shape_coords    : np.array,
                                shape_type      : int) -> np.array:
        """
        public fn: transform all coords of a given shapetype
        """
        shapetype_coords = self._find_shapetype_coords(shape_type=shape_type)
        return self._transform_coords(shapetype_coords)
    
    
    def find_distance_in_same_type(self):
        """
        transform coords
        select only the relevant shape type
        find the nearest object, recording the distance
        remove nearest object from list of available objects to search for next shapes
        """
        pass
    
    def _map_shape_types(self,
                         counts : dict) -> dict:

        counts ={4: 1, 2: 1}
        new_counts = {}
        
        for c_key, c_value in counts.items():
            for m_key, m_value in self.shape_type_mapping.items():
                if m_key == c_key:
                    new_counts[m_value] = c_value
        return new_counts
        
        
    
    def report_shape_match(self):
        """
        
        """
        our_solution = pd.Series(self.shape_types).value_counts() 
        known_solution = pd.Series( self.sol_summary[self.puzzle_id] )
        
        
        
        if our_solution != known_solution:
            print("Our solution had the following shapes:")
            print( self._map_shape_types(counts=our_solution.to_dict()))
        

#%%

instance = coord_trans(shape_coords = np.array( [[1,0,2],[1,2,3],[6,3,2],[1,2,2]] ),
                       shape_types = [1,2,3,1],
                       puzzle_id = '')

#instance.transform_all_shapetype(shape_type=1)

#%%

#instance.solution_summary
instance.report_shape_match()

#%%
instance.sol_pos


#%%
"""
1   find what shapes are used in the solution
2   find out how many of each shape are used
3   find out the distance to matching shape types
"""



#%%



    
#%%
                
instance = coord_trans(shape_coords = np.array( [[1,0,2],[1,2,3],[6,3,2],[1,2,2]] ),
            shape_types = [1,2,3,1])


print( instance.transform_coords(instance.shape_coords) )
print( instance._find_coord_centre() )


#%%
a = pd.DataFrame({'shapeType':[1,2,3,1], 'coords' : [ [1,0,2],[1,2,3],[6,3,2], [1,2,2]] })

for st in a.shapeType.unique():
    a.loc[a.shapeType == st, 'coords']




#%%
    
shapedata = solution_data.get('shapeData')




#%%





#%%
"""
list of all the possible interaction events
"""
print( df.type.unique() )



#%%
"""
To find all of the possible data fields:
"""
def find_all_pos_keys(df):
    key_combos = []
    all_keys = df.loc[:,'data'].apply(lambda x: list(json.loads(x).keys()) )
    
    
    unique_keys = []
    for i in range(len(df)):
        unique_keys.append(df.loc[i,'json_keys'])
        
    combined_list = []
    for i in unique_keys:
        combined_list += i
    return unique_keys

print( pd.Series(pd.Series(combined_list).unique()).sort_values().reset_index(drop=True) )

#%%

df.loc[:,'task_id'] = df.loc[:,'data'].apply(lambda x: json.loads(x).get('task_id'))

#%%


a = df.loc[df.type == 'ws-create_shape', 'data'].apply(lambda x: json.loads(x).keys())
a = a.reset_index(drop=True)
a[0]


#%%
"""
Find the most commonly created shapes
"""
a = df.loc[df.type == 'ws-create_shape', 'data'].apply(lambda x: json.loads(x).get('shapeType'))
print( a.value_counts() )


#%%

def find_all_pos_keys(df):
    all_keys = df.loc[:,'data'].apply(lambda x: list(json.loads(x).keys()) )
    unique_keys = []
    for i in range(len(df)):
        unique_keys.append(all_keys.loc[i])
    
    combined_list = []
    for i in unique_keys:
        combined_list += i
    
    return pd.Series(pd.Series(combined_list).unique()).sort_values().reset_index(drop=True)


unique_keys = find_all_pos_keys(df)
#for i in range(len(df)):
for key in unique_keys:
    try:
        df.loc[:,key] = df.loc[:,'data'].apply(lambda x: json.loads(x).get(key))
    except:
        pass

