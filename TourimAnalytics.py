import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, accuracy_score
import mysql.connector

# SQL query
sql = """
SELECT t.TransactionId, u.UserId, t.visityear, t.visitmonth, t.attractionid, t.rating, u.contenentid, u.regionid, u.countryid, u.cityid, r.region, c.country, m.visitmode,
i.attractioncityid, i.attractiontypeid, i.attraction, i.attractionaddress, ct.contenent, cty.cityname, tp.attractiontype
FROM transaction t
INNER JOIN user u ON t.userID = u.userID
INNER JOIN region r ON u.regionid = r.regionid
INNER JOIN country c ON c.countryid = u.countryid
INNER JOIN mode m ON m.visitmodeid = t.visitmode
INNER JOIN item i ON i.attractionid = t.attractionid
INNER JOIN continent ct ON ct.contenentid = u.contenentid
INNER JOIN city cty ON u.cityid = cty.cityid
INNER JOIN type tp ON tp.attractiontypeid = i.attractiontypeid
"""

# Cached function to load data
@st.cache_data
def load_data():
    connection = mysql.connector.connect(
        host='localhost', user='root', password='chennai', database='tourism',
        auth_plugin='mysql_native_password'
    )
    mycursor = connection.cursor()
    mycursor.execute(sql)
    data = mycursor.fetchall()
    column_names = [col[0] for col in mycursor.description]
    mycursor.close()
    connection.close()
    return pd.DataFrame(data, columns=column_names)

@st.cache_data
def preprocess_data(df):
    user_item_matrix = df.pivot_table('rating', ['UserId'], 'attractionid').fillna(0)
    user_mat_sim = cosine_similarity(user_item_matrix)
    return user_item_matrix, user_mat_sim

@st.cache_data
def encode_features(df):
    # Preserve original rating
    rating_col = df['rating']
    visitmode_col = df['visitmode']

    df = df.drop(['TransactionId', 'UserId', 'contenentid', 'regionid', 'countryid',
                  'cityid', 'attractiontypeid', 'attractioncityid', 'rating', 'visitmode'], axis=1)

    df_encoded = df.apply(LabelEncoder().fit_transform)
    df_encoded['rating'] = rating_col
    df_encoded['visitmode'] = LabelEncoder().fit_transform(visitmode_col)
    return df_encoded

@st.cache_resource
def train_models(X_train, y_train_clf, y_train_reg):
    model1 = RandomForestClassifier()
    model1.fit(X_train, y_train_clf)

    model2 = RandomForestRegressor()
    model2.fit(X_train, y_train_reg)

    return model1, model2

# Collaborative Filter function
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

# Collaborative Filter function for new users
def collabfillNew(country_id: int, city_id: int, visitmonth: int, df: pd.DataFrame) -> pd.DataFrame:
    filtered_users = df[
        (df['country'] == country_id) &
        (df['cityname'] == city_id) &
        (df['visitmonth'] == visitmonth)
    ]['TransactionId'].unique()
    #recommendations = df[df['TransactionId'].isin(recommended_ids)][['attraction', 'visitmode', 'rating']].drop_duplicates().head(5)
    recommendations = df[df['TransactionId'].isin(filtered_users)][['country','cityname','attractiontype','attraction', 'visitmode', 'rating']].drop_duplicates().head(5)
    return recommendations


# Load and preprocess data
transactiondetails_df = load_data()
user_recom = transactiondetails_df.copy()
user_item_matrix, user_mat_sim = preprocess_data(user_recom)
df_encoded = encode_features(transactiondetails_df.copy())

# Train/test split
X = df_encoded[["visityear", "visitmonth", "attractiontype"]]
y_reg = df_encoded["rating"]
y_clf = df_encoded["visitmode"]
X_train, X_test, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)
_, _, y_train_clf, y_test_clf = train_test_split(X, y_clf, test_size=0.2, random_state=42)

# Train or load models
model1, model2 = train_models(X_train, y_train_clf, y_train_reg)

# Streamlit UI
st.title("Tourism Prediction Analysis")

if "page" not in st.session_state:
    st.session_state.page = "Recommendation"

if st.session_state.page == "Recommendation":
    st.header("Recommendation - Existing user")
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

st.title('Recommendation - New User')
Country = st.text_input("Enter Country")
City = st.text_input("Enter City")
VisitMonth = st.number_input("Enter Visit Month", min_value=1, step=1)
if st.button("Recommendation - New User"):
    #st.write(transactiondetails_df.head(5))
    st.dataframe(collabfillNew(Country,City, VisitMonth, transactiondetails_df ))


st.title('Model Prediction App')
visityear = st.number_input('Enter visited year:')
visitmonth = st.number_input('Enter visited month:')
attractiontype = st.number_input('Enter attraction type:')
input_data = np.array([[visityear, visitmonth, attractiontype]])

if st.button("Predict Rating and Visit Mode"):
    rating_prediction = model2.predict(input_data)
    visit_mode_prediction = model1.predict(input_data)
    st.write(f'Predicted Rating: {rating_prediction[0]:.2f}')
    st.write(f'Predicted Visit Mode: {visit_mode_prediction[0]}')
