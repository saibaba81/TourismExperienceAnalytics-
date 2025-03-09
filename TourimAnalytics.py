import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import numpy as np
import base64
import mysql.connector
from mysql.connector import Error
from sqlalchemy import create_engine
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier




sql = """
SELECT t.TransactionId, u.UserId, t.visityear, t.visitmonth, t.attractionid, t.rating, u.contenentid, u.regionid, u.countryid, u.cityid, r.region, c.country, m.visitmode,
i.attractioncityid, i.attractiontypeid, i.attraction, i.attractionaddress, ct.contenent, cty.cityname, tp.attractiontype
FROM transaction t
INNER JOIN user u
ON t.userID = u.userID
inner join region r
on u.regionid = r.regionid
inner join country c
on c.countryid = u.countryid
inner join mode m
on m.visitmodeid = t.visitmode
inner join item i
on i.attractionid = t.attractionid
inner join continent ct
on ct.contenentid = u.contenentid
inner join city cty
on u.cityid = cty.cityid
inner join type tp
on tp.attractiontypeid = i.attractiontypeid

"""

#mysql database details
connection = mysql.connector.connect(host = 'localhost', user = 'root', password = 'chennai', database = 'tourism',auth_plugin='mysql_native_password')
mycursor = connection.cursor()

mycursor.execute(sql)
data = mycursor.fetchall()
column_names = [data[0] for data in mycursor.description]

mycursor.close()


transactiondetails_df = pd.DataFrame(data, columns = column_names)

df = transactiondetails_df
user_recom = transactiondetails_df
#user_recom = user_recom.sample(50000)

user_item_matrix = user_recom.pivot_table('rating', ['UserId'], 'attractionid')
user_item_matrix.fillna(0, inplace=True)
user_mat_sim = cosine_similarity(user_item_matrix)


# Collaborative Filter
def collabfill(user_id, user_mat_sim, user_item_matrix, user_recom):
    num_users = user_mat_sim.shape[0]

    if user_id < 1 or user_id > num_users:
        st.error(f"Error: User ID {user_id} is out of range. Valid range: 1 to {num_users}")
        return pd.DataFrame()

    simuser = user_mat_sim[user_id - 1]
    sim_user_ids = np.argsort(simuser)[::-1][1:6]
    sim_user_rating = user_item_matrix.iloc[sim_user_ids].mean(axis=0)
    rec_dest_id = sim_user_rating.sort_values(ascending=False).head(5).index
    rec = user_recom[user_recom['attractionid'].isin(rec_dest_id)][['attraction', 'visitmode', 'rating']].drop_duplicates().head(5)

    return rec


# Streamlit App
st.title("Tourism Prediction Analysis")


if "page" not in st.session_state:
    st.session_state.page = "Recommendation"  # Default to Recommendation


# Content Display based on Session State
if st.session_state.page == "Recommendation":
    st.header("Recommendation")
    with st.form(key='recommendation_form'):
        user_id = st.number_input("Enter User ID", min_value=1, step=1)
        submit_button = st.form_submit_button(label='Recommend')

    if submit_button:
        recom_data = collabfill(user_id, user_mat_sim, user_item_matrix, user_recom)
        if not recom_data.empty:
            st.subheader("Recommended Destinations:")
            st.dataframe(recom_data)
        else:
            st.warning("No recommendations found for this user.")

le = df.apply(LabelEncoder().fit_transform)
df.drop(['TransactionId'], axis = 1, inplace = True)
df.drop(['UserId'], axis = 1, inplace = True)
df.drop(['contenentid'], axis = 1, inplace = True)
df.drop(['regionid'], axis = 1, inplace = True)
df.drop(['countryid'], axis = 1, inplace = True)
df.drop(['cityid'], axis = 1, inplace = True)
df.drop(['attractiontypeid'], axis = 1, inplace = True)
df.drop(['attractioncityid'], axis = 1, inplace = True)


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score
X = le[["visityear", "visitmonth", "attractiontype"]]
y_reg = le["rating"]
y_clf = le["visitmode"]

X_train, X_test, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)
_, _, y_train_clf, y_test_clf = train_test_split(X, y_clf, test_size=0.2, random_state=42)

regressor = LinearRegression()
regressor.fit(X_train, y_train_reg)
y_pred_reg = regressor.predict(X_test)
print("Regression MSE:", mean_squared_error(y_test_reg, y_pred_reg))

classifier = RandomForestClassifier()
classifier.fit(X_train, y_train_clf)
y_pred_clf = classifier.predict(X_test)
print("Classification Accuracy:", accuracy_score(y_test_clf, y_pred_clf))

import pickle
from sklearn.ensemble import RandomForestClassifier
model1 = RandomForestClassifier()
# Assume X_train and y_train are your tr
model1.fit(X_train, y_train_clf)

# Save the model to a file
with open('model.pkl', 'wb') as file:
    pickle.dump(model1, file)


# Load the model
with open('model.pkl', 'rb') as file:
    model1 = pickle.load(file)


model2 = LinearRegression()
# Assume X_train and y_train are your training data
model2.fit(X_train, y_train_reg)

# Save the model to a file
with open('model.pkl', 'wb') as file:
    pickle.dump(model2, file)

# Load the model
with open('model.pkl', 'rb') as file:
    model2 = pickle.load(file)

st.title('Model Prediction App')
visityear = st.number_input('Enter visited year:')
visitmonth = st.number_input('Enter visited month:')

attractiontype = st.number_input('Enter attraction type:')
input_data = np.array([[visityear, visitmonth, attractiontype]])


# Prediction button
if st.button("Predict Rating"):
    #rating_prediction = model2.predict(input_data)
    visit_mode_prediction = model1.predict(input_data)
    #st.write(f'Predicted Rating: {rating_prediction[0]}')
    st.write(f'Predicted Visit Mode: {visit_mode_prediction[0]}')

