from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px
import altair as alt
import seaborn as sns
sns.set_style("whitegrid")
import base64
import datetime
from matplotlib import rcParams
from  matplotlib.ticker import PercentFormatter
import numpy as np
from sklearn.datasets import make_blobs
import plotly.express as px


@st.cache
def get_data():
  # return pd.read_csv("http://data.insideairbnb.com/united-states/ny/new-york-city/2019-09-12/visualisations/listings.csv")
  return pd.read_csv("data/listings.csv")


def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(sep='\t', decimal=',', index=False, header=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}">Download</a>'
    return href


def main():
  df = get_data()

  ################################ SIDEBAR ################################

  st.sidebar.header("Project for CSCI-657")

  st.sidebar.markdown(" ")
  st.sidebar.markdown("*This project is done as a part of the final project for the course Introduction to Data Mining*")

  st.sidebar.markdown("**Author**:Govind Pande, Ja-Yuan Pendley")
  #st.sidebar.markdown("**Mail**: gpande@nyit.edu")
  #st.sidebar.markdown("**NYIT ID**: 1302516")

  ################################ SUMMARY ################################

  st.title("Data Mining using Airbnb’s Open Dataset")
  st.markdown('-----------------------------------------------------')
  st.header("Introduction")

  st.markdown("Tourists looking to find accomodation often go to websites like Airbnb to find affordable accommodation. Airbnb provides various rental options for different customer segments. In this project we attempt to perform descriptive tasks on the Airbnb dataset provided by Airbnb. During the analysis of the dataset I performed tasks like association between location, price, number of reviews, availability, and hosts. Furthermore, K means clustering was used to cluster similar listings based on the given attributes. This helped us better understand the listings on Airbnb.")

  st.header("Dataset Description")

  st.markdown("The data is provided by Inside Airbnb dataset for New York City, New York, United States. **The dataSet can be found here: (http://insideairbnb.com/get-the-data/).** This dataset describes the listing activity and metrics in NYC, NY for 2021. The Features describe where people stay at most and what kind of place they want. These include data such as Names, neighbourhood_group, neighbourhood, room_type, and price. The complete dataset consists of 7244 records with 18 features or data itself 7244 rows and 18 columns.")
  st.markdown("The features in this dataset have 18 columns, and there is a mix between categorical and numeric values.")



  st.header("Project Summary")

  st.markdown("Airbnb is a platform that provides and guides the opportunity to link two groups - the hosts and the guests. Anyone with an open room or free space can provide services on Airbnb to the global community. It is a good way to provide extra income with minimal effort. It is an easy way to advertise space, because the platform has traffic and a global user base to support it. Airbnb offers hosts an easy way to monetize space that would be wasted.")

  st.markdown("On the other hand, we have guests with very specific needs - some may be looking for affordable accommodation close to the city's attractions, while others are a luxury apartment by the sea. They can be groups, families or local and foreign individuals. After each visit, guests have the opportunity to rate and stay with their comments. We will try to find out what contributes to the listing's popularity and predict whether the listing has the potential to become one of the 100 most reviewed accommodations based on its attributes.")

  st.markdown('-----------------------------------------------------')

  st.header("Airbnb New York Listings: Data Analysis")
  st.markdown("Following is presented the first 10 records of Airbnb data. These records are grouped along 16 columns with a variety of informations as host name, price, room type, minimum of nights,reviews and reviews per month.")
  st.markdown("We will start with familiarizing ourselves with the columns in the dataset, to understand what each feature represents. This is important, because a poor understanding of the features could cause us to make mistakes in the data analysis and the modeling process. We will also try to reduce number of columns that either contained elsewhere or do not carry information that can be used to answer our questions.")

  st.dataframe(df.head(10))

  st.markdown("Another point about our data is that it allows sorting the dataframe upon clicking any column header, it a more flexible way to order data to visualize it.")

  #################### DISTRIBUIÇÃO GEOGRÁFICA ######################

  st.header("Listing Locations")
  st.markdown("Airbnb’s first New York listing was in Harlem in the year 2008, and the growth since has been exponential. Below we highlight the geographical distribution of listings. Initially we can filter them by price range, minimum number of available nights and number of reviews, so more flexibility is added when looking for a place. ")
  st.markdown("We could also filter by listing **price**, **minimum nights** on a listing or minimum of **reviews** received. ")


  Min_price = float(st.number_input('Min Price', float((df.price.min())), float((df.price.max())), 10.0 , step =10.0))
  Max_price = float(st.number_input('Max Price', float((df.price.min())), float((df.price.max())), 5000.0, step=10.0))
  #min_value = st.slider("Min Price ($)", (df.price.min()), (df.price.clip(upper=10000.).max()), (500., 1500.))
  min_nights_values = st.slider('Minimum Nights', 0, 30, (30))
  reviews = st.slider('Minimum Reviews', 0, 700, (0))
  st.map(df.query(f"price<={Max_price} and price>={Min_price} and  minimum_nights<={min_nights_values} and number_of_reviews>={reviews}")[["latitude", "longitude"]].dropna(how="any"), zoom=10)
  #st.map(df[["latitude", "longitude"]], zoom=10)
  st.markdown("In a general way the map shows that locations in the city centre are more expensive, while the outskirts are cheaper (a pattern that probably does not only exists in New York). In addition, the city centre seems to have its own pattern.")
  st.markdown("Unsurprisingly, Manhattan island has the highest concentration of expensive Airbnbs. Some are scattered over Brooklyn too. The heftiest price tag is $10.000,00. Another likely insight is that if we know that a specific location is very close to a place we consider expensive most probably the whole sorrounding area will be expensive.")
  st.markdown("Highly rated locations also tend to be the most expensive ones. Again downtown Manhattan and adjacent areas of Brooklyn receive the highest location scores, with East Village being an exception. A marked drop in location scores is seen as the subway lines end.")
  st.markdown("In a side analysis it can be possible to see that around Manhattan there are much fewer flats than compared to areas around, in addition, most of the points of interest (_Empire State Buildind, Times Square, Central Park_) are located in ‘expensive’ areas, especially around Dam Square's district.")
  st.markdown("In Staten Island, the areas close to the State Park have the highest location scores. Brooklyn neighbourhoods close to Manhattan tend to have higher location ratings. Looking at the NY subway system in Brooklyn, it is interesting to observe that the highly rated areas correspond with subway line presence. The same is true for Bronx where subway lines do not go.")

  #################### AREAS OF INTEREST ######################

  st.header("Explore the dataset!")
  st.write(f"Out of the {df.shape[1]} columns, you can select columns of your interest in the dataset. These are the most correlated columns to the price in listing")
  defaultcols = ["price", "minimum_nights", "room_type", "neighbourhood", "name", "number_of_reviews"]
  cols = st.multiselect('', df.columns.tolist(), default=defaultcols)
  st.dataframe(df[cols].head(10))


  ################################## DISTRICT ###############################

  st.header("Districts")
  st.markdown("The New York City encompasses five county-level administrative divisions called boroughs namely Bronx , Brooklyn, Manhattan, Queens and Staten Island. ")

  st.markdown("Again unsurprisingly it is possible see that the average price in the Manhattan district can be much higher than other districts. Manhattan has an average price of twice the Bronx ")

  fig = sns.barplot(x='neighbourhood_group', y='price', data=df.groupby('neighbourhood_group')['price'].mean().sort_values(ascending=False).reset_index(),
  palette="Blues_d")
  sns.set(font_scale = 1.5)
  fig.set_xlabel("District",fontsize=10)
  fig.set_ylabel("Price ($)",fontsize=10)
  st.pyplot()


  st.markdown("Below we can see the density distribution against the price for each borough in the city. I got rid of the properties with price more than 500$ for the sake of this visualization. ")


  listings = pd.read_csv("data/listings.csv")
  #listings = listings["price"].dropna()
  #creating a sub-dataframe with no extreme values / less than 500
  sub_6=listings[listings.price < 500]
  #using violinplot to showcase density and distribtuion of prices 
  viz_2=sns.violinplot(data=sub_6, x='neighbourhood_group', y='price')
  viz_2.set_title('Density and distribution of prices for each neighberhood_group')
  st.pyplot()


  ################### PERCENTAGE DISTRIBUTION BY DISTRICT #####################

  st.header("Availability and Distribution by District.")
  st.markdown("The **availability_365** feature mean the number of days of the year (365) listing availability. Let's check it out.")

  neighborhood = st.radio("District", df.neighbourhood_group.unique())
  is_expensive = st.checkbox("Expensive Listings")
  is_expensive = " and price<100" if not is_expensive else ""

  @st.cache
  def get_availability(show_exp, neighborhood):
      return df.query(f"""neighbourhood_group==@neighborhood{is_expensive}\
          and availability_365>0""").availability_365.describe(\
              percentiles=[.1, .25, .5, .75, .9, .99]).to_frame().T

  st.table(get_availability(is_expensive, neighborhood))
  st.markdown("_**Note:** There are 4187 records with **availability_365** 0 (zero), which I've ignored._")
  st.markdown("At 185 days, Manhattan has the lowest average availability. At 226, Queens has the highest average availability. If we include expensive listings (more tha $100 in a listing), the numbers are 184 for Manhattan and 241 for Staten Island.")

  ###################### QUANTITY OF ROOM TYPES BY DISTRICT #######################

  st.markdown("Following, let's check the relationship between property type and neighbourhood. The primary question we aim to answer is whether different boroughs constitute of different rental types. Though in the expanded dataset there are more than 20 types, we will be focussing on the top 4 by their total count in the city and understanding their distribution in each borough.")

  room_types_df = df.groupby(['neighbourhood_group', 'room_type']).size().reset_index(name='Quantity')
  room_types_df = room_types_df.rename(columns={'neighbourhood_group': 'District', 'room_type':'Room Type'})
  room_types_df['Percentage'] = room_types_df.groupby(['District'])['Quantity'].apply(lambda x:100 * x / float(x.sum()))

  sns.set_style("darkgrid")
  sns.set(rc={'figure.figsize':(11.7,8.27)})
  fig = sns.catplot(y='Percentage', x='District', hue="Room Type", data=room_types_df, height=6, kind="bar", palette="muted", ci=95);
  fig.set(ylim=(0, 100))


  for ax in fig.axes.flat:
      ax.yaxis.set_major_formatter(PercentFormatter(100))
  plt.show()

  st.pyplot()

  st.markdown("The plot shows the ratio of property type and the total number of properties in the borough.")

  st.subheader("Some key observations from the graph are:")

  st.markdown(" - We can see that **Entire home/apt** listings are highest in number in all tree borough except Bronx and Queens. Bronx has more ‘Private room’ style property than ‘Apartments’.")

  st.markdown(" - The maximum **Entire home/apt** listings are located in Manhattan, constituting 60.55% of all properties in that neighborhood. Next is Staten Island with 49.86% **Entire home/apt**.")

  st.markdown(" - Queens and the Bronx also have many listings for **Private room**. In the Bronx, 60% of the apartments are of the **Private room** type, which is larger than in the Queens.")

  st.markdown(" - **Shared Room** and **Hotel rooms** listing types are not very common in New York.")

  ###################### PRICE AVERAGE BY ACOMMODATION #########################

  st.header("Average price by room type")

  st.markdown("To listings based on room type, we can show price average grouped by borough.")

  avg_price_room = df.groupby("room_type").price.mean().reset_index()\
      .round(2).sort_values("price", ascending=False)\
      .assign(avg_price=lambda x: x.pop("price").apply(lambda y: "%.2f" % y))

  avg_price_room = avg_price_room.rename(columns={'room_type':'Room Type', 'avg_price': 'Average Price ($)', })

  st.table(avg_price_room)

  st.markdown("Despite together **Hotel Room** listings represent just over 10%, they are responsible for the highest price average, followed by **Entire home/apt**. Thus there are a small number of **Hotel Room** listings due its expensive prices.")


  ############################ MOST RATED HOSTS #############################

  st.header("Most rated hosts")

  rcParams['figure.figsize'] = 15,7
  ranked = df.groupby(['host_name'])['number_of_reviews'].count().sort_values(ascending=False).reset_index()
  ranked = ranked.head(5)
  sns.set_style("whitegrid")
  fig = sns.barplot(y='host_name', x='number_of_reviews', data=ranked,palette="Blues_d",)
  fig.set_xlabel("Nº de Reviews",fontsize=10)
  fig.set_ylabel("Host",fontsize=10)

  st.pyplot()

  st.write(f"""The host **{ranked.iloc[0].host_name}** is at the top with {ranked.iloc[0].number_of_reviews} reviews.
  **{ranked.iloc[1].host_name}** is second with {ranked.iloc[1].number_of_reviews} reviews. It should also be noted that reviews are not positive or negative reviews, but a count of feedbacks provided for the accommodation.""")


  top_host=listings.host_id.value_counts().head(10)
  top_host_df=pd.DataFrame(top_host)
  top_host_df.reset_index(inplace=True)
  top_host_df.rename(columns={'index':'Host_ID', 'host_id':'P_Count'}, inplace=True)
  #top_host_df



  sns.set(rc={'figure.figsize':(15,7)})
  #sns.set_style('white')

  viz_1=sns.barplot(x="Host_ID", y="P_Count", data=top_host_df,
                 palette='Blues_d')
  viz_1.set_title('Hosts with the most listings in NYC')
  viz_1.set_ylabel('Count of listings')
  viz_1.set_xlabel('Host IDs')
  viz_1.set_xticklabels(viz_1.get_xticklabels(), rotation=45)

  st.pyplot()


  st.write(f"""The host with host id **7503643** is at the top with **29** properties.
  **417504** is second with **27** reviews. It should also be noted that reviews are not positive or negative reviews, but a count of feedbacks provided for the accommodation.""")





  #################### DEMAND AND PRICE ANALYIS ######################

  st.header("Demand and Price Analysis")

  st.markdown("In this section, we will analyse the demand for Airbnb listings in New York City. We will look at demand over the years since the inception of Airbnb in 2010 and across months of the year to understand seasonlity. We also wish to establish a relation between price and demand. The question we aspire to answer is whether prices of listings fluctuate with demand. We will also conduct a more granular analysis to understand how prices vary by days of the week.")
  st.markdown("To study the demand, since we did not have data on the bookings made over the past year, we will use **number of reviews** variable as the indicator for demand. As per Airbnb, about 50% of guests review the hosts/listings, hence studying the number of review will give us a good estimation of the demand.")

  accommodation = st.radio("Room Type", df.room_type.unique())

  all_accommodation = st.checkbox('All Accommodations')

  demand_df = df[df.last_review.notnull()]
  demand_df.loc[:,'last_review'] = pd.to_datetime(demand_df.loc[:,'last_review'])
  price_corr_df = demand_df

  if all_accommodation:
    demand_df = df[df.last_review.notnull()]
    demand_df.loc[:,'last_review'] = pd.to_datetime(demand_df.loc[:,'last_review'])
  else:
    demand_df = demand_df.query(f"""room_type==@accommodation""")

  fig = px.scatter(demand_df, x="last_review", y="number_of_reviews", color="room_type")
  fig.update_yaxes(title="Nª Reviews")
  fig.update_xaxes(title="Last Review Dates")
  st.plotly_chart(fig)

  st.markdown("The number of unique listings receiving reviews has increased over the years. Highly rated locations also tend to be the most expensive ones. We can see an almost exponential increase in the number of reviews, which as discussed earlier, indicates an exponential increase in the demand.")

  st.markdown("But about the price ? We also can show the same plot, but this time we take into account the **price** feature along years. Again we use **last review dates** to modeling time series in order to achieve a proportion between price over the years. Let's check it out.")

  fig = px.scatter(price_corr_df, x="last_review", y="price", color="neighbourhood_group")
  fig.update_yaxes(title="Price ($)")
  fig.update_xaxes(title="Last Review Dates")
  st.plotly_chart(fig)

  st.markdown("The price smoothly increases along the years if we take into account the number of reviews according the borough. Manhattan is most expensive borough followed by Brooklyn, some listings also appear as outliers in the year 2015.\nLet's take a look again in the number of reviews, but this time we group by boroughs to give us an idea of the distribution of reviews among the boroughs.")

  fig = px.scatter(price_corr_df, x="last_review", y="number_of_reviews", color="neighbourhood_group")
  fig.update_yaxes(title="Nª Reviews")
  fig.update_xaxes(title="Last Review Dates")
  st.plotly_chart(fig)

  st.markdown("The number of reviews for Queens appears more often. We get some insights here. \n1) The room type most sought in Queens is the **private room** (as seen in the previous plot). \n2) The price range in Queens is below Manhattan, so perhaps the Queens contemplate the _\"best of both worlds\"_ being the most cost-effective district.")

  st.markdown("But there is some correlation between reviews increase and prices? Let's check it out.")

  fig = px.scatter(price_corr_df, y="price", x="number_of_reviews", color="neighbourhood_group")
  fig.update_xaxes(title="Nª Reviews")
  fig.update_yaxes(title="Price ($)")
  st.plotly_chart(fig)

  st.markdown("There doesn't seem to be any correlation between the reviews and the prices. We do see that the cheaper the price the more reviews a property has. This also reflects that there are simply fewer expensive properties when compared to moderate or low priced properties. We also see that Queens has more reviews than other boroughs which reinforces our theory that Queens is the most value for money borough.")


  st.header("Most Rated Listings")
  st.markdown("We can slide to filter a range of numbers in the sidebar to view properties whose review count falls in that range.")

  reviews = st.slider('', 0, 12000, (100))

  df.query(f"number_of_reviews<={reviews}").sort_values("number_of_reviews", ascending=False)\
  .head(50)[["number_of_reviews", "price", "neighbourhood", "room_type", "host_name"]]

  st.write("674 is the highest number of reviews and only a single property has it. In general, listings with more than 400 reviews are priced below $ 100. Some are between $100,00 and $200 and only one is priced above $200.")


  ############################# PRICE DISTRIBUTION ###########################

  st.header("Price Distribution")


  Min_price1 = float(st.number_input('Min Price', float((df.price.min())), float((df.price.max())), 10.0))
  Max_price1 = float(st.number_input('Max Price', float((df.price.min())), float((df.price.max())), 5000.0))
  #st.map(df.query(f"price<={Max_price} and price>={Min_price} and  minimum_nights<={min_nights_values} and number_of_reviews>={reviews}")[["latitude", "longitude"]].dropna(how="any"), zoom=10)

  st.markdown("Bellow we can select a custom price range from the side bar to update the histogram below and check the distribution skewness.")
  st.write("""Select a custom price range from the side bar to update the histogram below.""")
  #values = st.slider("Faixa de Preço", float(df.price.min()), float(df.price.clip(upper=1000.).max()), (50., 300.))
  f = px.histogram(df.query(f"price<={Max_price} and price>={Min_price} "), x="price", nbins=100, title="Price distribution")
  f.update_xaxes(title="Price")
  f.update_yaxes(title="No. of listings")
  st.plotly_chart(f, color='lifeExp')

  @st.cache
  def get_availability(show_exp, neighborhood):
    return df.query(f"""neighbourhood_group==@neighborhood{show_exp}\
    and availability_365>0""").availability_365.describe(\
    percentiles=[.1, .25, .5, .75, .9, .99]).to_frame().T




  ################################## K MEANS ##################################
  st.header("K Means Clustering")

  st.markdown("For the given dataset, we chose the number of equal to 5. However, users of this application can select different values for number of clusters to experiment.")
  num_clusters = st.number_input('Please select the number of clusters', 1, 10, 5) # num of cluster
  class KMeansClustering:
      def __init__(self, X, num_clusters):
          self.K = num_clusters # cluster number
          self.max_iterations = 100 # max iteration. don't want to run inf time
          self.num_examples, self.num_features = X.shape # num of examples, num of features
          self.plot_figure = True # plot figure
        
    # randomly initialize centroids
      def initialize_random_centroids(self, X):
          centroids = np.zeros((self.K, self.num_features)) # row , column full with zero 
          for k in range(self.K): # iterations of 
              centroid = X[np.random.choice(range(self.num_examples))] # random centroids
              centroids[k] = centroid
          return centroids # return random centroids
    
    # create cluster Function
      def create_cluster(self, X, centroids):
          clusters = [[] for _ in range(self.K)]
          for point_idx, point in enumerate(X):
              closest_centroid = np.argmin(
                  np.sqrt(np.sum((point-centroids)**2, axis=1))
              ) # closest centroid using euler distance equation(calculate distance of every point from centroid)
              clusters[closest_centroid].append(point_idx)
          return clusters 
    
    # new centroids
      def calculate_new_centroids(self, cluster, X):
          centroids = np.zeros((self.K, self.num_features)) # row , column full with zero
          for idx, cluster in enumerate(cluster):
              new_centroid = np.mean(X[cluster], axis=0) # find the value for new centroids
              centroids[idx] = new_centroid
          return centroids
    
    # prediction
      def predict_cluster(self, clusters, X):
          y_pred = np.zeros(self.num_examples) # row1 fillup with zero
          for cluster_idx, cluster in enumerate(clusters):
              for sample_idx in cluster:
                  y_pred[sample_idx] = cluster_idx
          return y_pred
    
    # plotinng scatter plot
      def plot_fig(self, X, y):
          fig = px.scatter(X[:, 0], X[:, 1], color=y)
          #fig.show() # visualize
          #st.plotly(fig)
          st.plotly_chart(fig)
          
        
    # fit data
      def fit(self, X):
          centroids = self.initialize_random_centroids(X) # initialize random centroids
          for _ in range(self.max_iterations):
              clusters = self.create_cluster(X, centroids) # create cluster
              previous_centroids = centroids
              centroids = self.calculate_new_centroids(clusters, X) # calculate new centroids
              diff = centroids - previous_centroids # calculate difference
              if not diff.any():
                  break
          y_pred = self.predict_cluster(clusters, X) # predict function
          if self.plot_figure: # if true
              self.plot_fig(X, y_pred) # plot function 
          return y_pred
            
  if __name__ == "__main__":
      np.random.seed(10)
      #num_clusters = st.sidebar.number_input('Please select the number of clusters', 1, 10, 5) # num of cluster
      X, _ = make_blobs(n_samples=7233, n_features=5, centers=num_clusters) # create dataset using make_blobs from sklearn datasets
      Kmeans = KMeansClustering(X, num_clusters)
      y_pred = Kmeans.fit(X)
      #st.plotly_chart(y_pred)


  st.markdown("For creating the clusters, we used 5 features from the dataset. The features are neighbourhood group, neighborhood, property type, price and number of reviews. The above visualization illustrates the distribution of clusters for the selected features.")
  

        







  ############################# CONCLUSIONS ###########################

  st.header("Conclusions")

  st.markdown("Through this exploratory data mining and visualization project, we gained several interesting insights into the Airbnb rental market. Below we will summarise the answers to the questions that we wished to answer at the beginning of the project:")

  st.markdown("**How do prices of listings vary by location? What localities in NYC are rated highly by guests?** Manhattan has the most expensive rentals compared to the other boroughs. Prices are higher for rentals closer to city hotspots. Rentals that are rated highly on the location by the host also have higher prices")

  st.markdown("**How does the demand for Airbnb rentals fluctuate across the year and over years?** In general, the demand (assuming that it can be inferred from the number of reviews) for Airbnb listings has been steadily increasing over the years.")

  st.markdown("**Are the demand and prices of the rentals correlated?** Average prices of the rentals increase across the year, which correlates with demand.")

  st.markdown("**Are the number of reviews and prices of the rentals correlated?** There doesn't seem to be any correlation between the reviews and the prices. We do see that the cheaper the price the more reviews a property has. This also reflects that there are simply fewer expensive properties when compared to moderate or low priced properties. We also see that Queens has more reviews than other boroughs which reinforces our theory that Queens is the most value for money borough.")


  st.header("Future Works")

  st.markdown("The calender dataset provided by Airbnb contains the price data for different dates on the calender. Analyzing that dataset will provide useful insights on how the prices of the listings change every week or month. If these fluctuations are corelated with the weather, holidays, months, etc. We did not have data for past years and hence could not compare current rental trends with past trends. Hence, there was an assumption made, particularly in the demand and supply section of the report to understand the booking trends. Future work can done with both of the datasets can provide more insight on this. ")


  ################################## FOOTER ##################################

  st.markdown('-----------------------------------------------------')
  st.text('Developed by Govind Pande & Ja-Yuan Pendley - 12-12 2022')

st.set_option('deprecation.showPyplotGlobalUse', False)

if __name__ == '__main__':
  main()
