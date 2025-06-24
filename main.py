import streamlit as st
from streamlit_navigation_bar import st_navbar
import pages as pg

st.set_page_config(layout="wide")

# def login():
#     st.markdown("<h2 style='text-align: center;'>Login</h2>", unsafe_allow_html=True)

#     col1, col2, col3 = st.columns([1, 2, 1])  # Adjust column width as needed

#     with col2:  # Centered smaller input boxes
#         username = st.text_input("Username", key="username")
#         password = st.text_input("Password", type="password", key="password")
    
#     st.markdown("<br>", unsafe_allow_html=True)  

#     col1, col2, col3 = st.columns([19, 3, 20])
#     with col2:
#         logger = st.button("Login", use_container_width=True) 
#     if logger:
#         if username == USERNAME and password == PASSWORD:
#             st.session_state.logged_in = True
#             st.success("Login Successful!")
#         else:
#             st.error("Invalid credentials")

# if 'logged_in' not in st.session_state:
#     st.session_state.logged_in = False

# if not st.session_state.logged_in:
#     login()


pages = ["Surveillance", "Fall Detection", "Register"]

# styles = {
#     "nav": {
#         "background-color": "royalblue",
#         "justify-content": "left",
#     },
#     "img": {
#         "padding-right": "14px",
#     },
#     "span": {
#         "color": "white",
#         "padding": "14px",
#     },
#     "active": {
#         "background-color": "white",
#         "color": "var(--text-color)",
#         "font-weight": "normal",
#         "padding": "14px",
#     }
# }

page = st_navbar(
    pages,
    # styles=styles
    selected = "Fall Detection"
)

functions = {
    "Fall Detection": pg.show_falldetection,
    "Surveillance": pg.show_surveillance,
    "Register": pg.show_register,
}
go_to = functions.get(page)
if go_to:
    go_to()



# Login form

