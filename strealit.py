import streamlit as st
st.image('./logo.png')
st.title('Head the full')

import pandas as pd

df = pd.read_csv('./spam.csv')
#df.head()

df.groupby('Category').describe()

df['spam'] = df['Category'].apply(lambda x: 1 if x== 'spam' else 0)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df.Message, df.spam, test_size = 0.2)

from sklearn.feature_extraction.text import CountVectorizer
v = CountVectorizer()
x_train_count = v.fit_transform(x_train.values)
x_train_count.toarray()

from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(x_train_count, y_train)

x_test_count = v.transform(x_test)


# streamlit work
st.sidebar.header('The built-in user input messages.')
user_input_1 = st.sidebar.selectbox('Messages',('how are you kanhaiya', 'flat 50% off on your first purchase.', 'you have exhausted your data ', 'hello mohan want to go out for play', 'i will call you later mummy', 'order the burger and get 20% discount'))

st.header('* The spam detection app')
user_input = st.text_input('Enter the e-mail/message here', user_input_1)
user_input = [user_input]
email_count = v.transform(user_input)
pre = model.predict(email_count)

if pre == 0:
    st.subheader('* it is not a spam.')
elif pre == 1:
    st.subheader('* it is a spam.')
else:
    None

x = model.score(x_test_count, y_test)
st.write('The accuracy of the model is',model.score(x_test_count, y_test))
if (x > 0.95):
    st.subheader('Probability is VERY HIGH ')

