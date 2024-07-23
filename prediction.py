import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import sklearn.model_selection
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
import matplotlib.pyplot as plt
import plotly.graph_objects as go

with st.sidebar:
    uploaded_file = st.file_uploader("Upload Data File",type='CSV')
tab1,tab2 = st.tabs(['Train', 'Inference'])
if uploaded_file is not None:
    with tab1:
        col1,col2 = st.columns(2)
        with col1:
            df = pd.read_csv(uploaded_file)
            st.dataframe(df)
        with col2:
            options = st.multiselect('Select features', df.columns[:-1])
            if options:
                X = df[options]
                name = ', '.join(options)
                y = df[df.columns[-1]]
                X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.3)
                model = LinearRegression()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                st.info(f'Model trained, MAE:{mae(y_test, y_pred):.3f}, MSE:{mse(y_test, y_pred):.3f}')
                if len(options) == 1:
                    fig, ax = plt.subplots()
                    x_line = [X.min(),X.max()]
                    y_line = model.predict(x_line)
                    ax.plot(x_line,y_line,c='r')
                    ax.scatter(X,y)
                    ax.set_xlabel(name)
                    ax.set_ylabel(df.columns[-1])
                    st.pyplot(fig)
                if len(options) == 2:
                    X = np.array(X)
                    x_line = np.linspace(X[:,0].min(), X[:,0].max(), 100)
                    y_line = np.linspace(X[:,1].min(), X[:,1].max(), 100)
                    xx,yy = np.meshgrid(x_line, y_line)
                    X_line = np.c_[xx.ravel(), yy.ravel()]
                    z_line = model.predict(X_line).reshape(xx.shape)
                    fig = go.Figure(data=[go.Scatter3d(x=X[:,0],y=X[:,1],z=y,mode='markers'),
                                          go.Surface(x=xx,y=yy,z=z_line,name='Regression Plane')])
                    fig.update_layout(
                        scene=dict(
                            xaxis_title=options[0],
                            yaxis_title=options[1],
                            zaxis_title=df.columns[-1]))
                    st.plotly_chart(fig)
                else:
                    st.warning('Cannot graph')
    with tab2:
        col = st.columns(len(options))
        x = []
        for i in range(len(options)):
            with col[i]:
                n = st.number_input(options[i],0.0)
            x.append(n)
        if x == [0 for i in range(len(options))]:
            names = name.replace(',',' or ')
            st.warning(f'Please input: {names}')
        else:
            x = np.array([x])
            y = model.predict(x)
            st.success(f'Prediction: {y[0]:.4f}')