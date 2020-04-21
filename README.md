# Machine Learning Assignment 03

Option 02:  
Clustering image meta-data

## Process
### Data work:
I chose to cluster the images based on the following datapoints:
'primary_medium',
'art_movement',
'representation',
'has_text', and
'spatial_dimension'.  

I replaced all NaN values with zeros:
```
data['representation'] = data['representation'].replace(np.nan, 0)
```
And further replaced strings (in 'primary_medium'
an 'art_movement') by assigning them a unique integer:
```
# perform value count
vc_am = am['art_movement'].value_counts()
categories = vc_am.to_frame()

# access index values and create list
iv_am = categories.index.values.tolist()
print(iv_am)

# replace values with integers
def category_to_numeric(x):
    i = 0
    while i<len(iv_am):
        if x==iv_am[i]:
            return i
        i = i+1

# apply function and add new column to X
X['art_movement'] = am['art_movement'].apply(category_to_numeric)
```
### Feature transformations:
Feature scaling with MinMaxScaler:
```
mm = MinMaxScaler(feature_range=(0, 5), copy=True)
mm_X = mm.fit_transform(X)
```
### Fitting K-Means
The inertia plot shows an elbow around n=10: 

![inertia plot](/img/inertia_scores.png)  

The average silhouette score peaks at n=9 with 0.645632800066643, which is why I chose n=9 as my final fit for k-means.  
Following the silhoutte plot:

![silhouette plot](/img/silhouette_plot.png)

Please reference the jupyter notebook to see the clustered images.
