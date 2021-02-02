import dash_core_components as dcc
import plotly
import plotly.graph_objs as go
from collections import deque
import pandas as pd
import numpy as np 

import json
import dash_html_components as html
import dash
from dash.dependencies import Input, Output
from dash_extensions import WebSocket
from dash_extensions.websockets import SocketPool, run_server
import plotly.express as px

class Plot:
    yellow = "rgb(255, 200, 87)"
    blue = "rgb(37, 110, 255)"
    green = "rgb(61, 220, 151)"
    red = "rgb(255, 73, 92)"
    purple = "rgb(66, 32, 64)"
    
    def __init__(self, callback, first_img):
        self.traj_x = deque(maxlen=200)
        self.traj_y = deque(maxlen=200)
        self.traj_z = deque(maxlen=200)
        self.traj_x.append(0)
        self.traj_y.append(0)
        self.traj_z.append(0)

        self.real_traj_x = deque(maxlen=200)
        self.real_traj_y = deque(maxlen=200)
        self.real_traj_z = deque(maxlen=200)
        self.real_traj_x.append(0)
        self.real_traj_y.append(0)
        self.real_traj_z.append(0)

        self.err_x = deque(maxlen=100)
        self.err_y = deque(maxlen=100)
        self.err_angle = deque(maxlen=100)
        self.err_x.append(0)
        self.err_y.append(0)
        self.err_angle.append(0)

        self.point_cloud = deque(maxlen=5000)
        self.point_cloud.append([0,0,0])

        self.video=[]

        self.points=[]

        self.fig = go.Figure()
        
        
        self.app = dash.Dash(prevent_initial_callbacks=True)
        self.socket_pool = SocketPool(self.app, handler=callback)
        self.app.layout = html.Div(
            [   
                dcc.Markdown(
                    '''
                    # Visual Odometry mini-Project Vincenzo Polizzi, 2020
                    Graphical representation of the visual odometry results
                    for more information visit [github.com](https://github.com/viciopoli01/VisualOdometryWithSemantic)
                    ''') ,

                dcc.Markdown('''
                    ## Camera stream.
                    ''') ,
                dcc.Graph(id='image_stream',
                    figure=px.imshow(first_img)
                ),
                dcc.Markdown('''
                    ## Estimated and real Duckiebot pose, point cloud.
                    ''') ,
                dcc.Graph(id='traj-graph', figure=go.Figure(
                    data=go.Scatter3d(
                        x=list(self.traj_x),
                        y=list(self.traj_z),
                        z=list(self.traj_y),
                        mode='lines',
                        marker=dict(size=5, color=self.red),
                        line=dict(
                            color=self.red,
                            width=5
                        ),
                        visible=True,
                        text="Trajectory",
                        name="Trajectory",
                        hoverinfo='text',
                        showlegend=True
                        )
                    )
                ),
                dcc.Markdown('''
                    ## Errors along x,y and on the rotation angle.
                    ''') ,
                dcc.Graph(id='err-graph', figure=go.Figure(
                    data=go.Scatter(
                        x=list(self.err_x), 
                        y=list([0]),
                        mode='lines',
                        name='lines'
                        )
                    ),
                ),
                WebSocket(id="ws")
            ]
        )
        #Output("data", "children") 
        self.app.callback(Output('traj-graph', 'figure'), Output('err-graph', 'figure'),Output('image_stream', 'figure'),
        [Input('ws', 'message')])(self.update_graph_scatter)
        
    def start_server(self):
        run_server(self.app, 5000)
        #run_server(debug=True,host='127.0.0.1', port=8011)

    def update_graph_scatter(self, msg):
        cloud = pd.DataFrame(data=self.point_cloud, columns=['x', 'y', 'z'])
        # print(cloud)
        data_traj = [
            go.Scatter3d(
                x=list(self.traj_x),
                y=list(self.traj_y),
                z=list(self.traj_z),
                mode='lines+markers',
                marker=dict(size=5, color=self.red),
                line=dict(
                    color=self.red,
                    width=5
                ),
                visible=True,
                text="Trajectory",
                name="Trajectory",
                hoverinfo='text',
                showlegend=True
            ),
            go.Scatter3d(
                x=list(self.real_traj_x),
                y=list(self.real_traj_y),
                z=list(self.real_traj_z),
                mode='lines+markers',
                marker=dict(size=5, color=self.green),
                line=dict(
                    color=self.green,
                    width=5
                ),
                visible=True,
                text="Real Trajectory",
                name="Real Trajectory",
                hoverinfo='text',
                showlegend=True
            ),
            go.Scatter3d(
                x=cloud['x'],
                y=cloud['z'],
                z=-cloud['y'],
                mode='markers',
                marker=dict(size=5, color=self.blue),
                visible=True,
                text="Features cloud",
                name="Features cloud",
                hoverinfo='text',
                showlegend=True
            )]
        steps = len(self.err_x)
        steps=list(np.linspace(1, steps, num=steps))
        data_err=[
            go.Scatter(
                x=steps, 
                y=list(self.err_x),
                mode='lines',
                name='Error along x',
                visible=True,
                hoverinfo='text',
                marker=dict(size=5, color=self.blue),
                line=dict(
                    color=self.blue,
                    width=5
                )
            ),
            go.Scatter(
                x=steps, 
                y=list(self.err_y),
                mode='lines',
                name='Error along y',
                visible=True,
                hoverinfo='text',
                marker=dict(size=5, color=self.green),
                line=dict(
                    color=self.green,
                    width=5
                )
            ),
            go.Scatter(
                x=steps, 
                y=list(self.err_angle),
                mode='lines',
                name='Error on the angle',
                visible=True,
                hoverinfo='text',
                marker=dict(size=5, color=self.purple),
                line=dict(
                    color=self.purple,
                    width=5
                )
            )]
        
        return [self.__graph_data(data_traj,[min(list(self.real_traj_x)),max(list(self.real_traj_x))],[min(list(self.real_traj_y)),max(list(self.real_traj_y))]),
                    self.__graph_data(data_err,[min(steps),max(steps)], [min(list(self.err_x)),max(list(self.err_x))]),
                    px.imshow(self.video)
                ]
    
    def __graph_data(self,data,min_max_x,min_max_y):
        return {'data': data, 
                'layout' : go.Layout(xaxis=dict(range=min_max_x), yaxis=dict(range=min_max_y),scene=dict(aspectmode="cube",aspectratio=dict(x=1, y=1, z=1))),
                'height':1000}

    def draw_line(self):
        self.fig.add_trace(self.__pointsShape(self.points, "Traj", "Traj", self.green))

    def add_traj_point(self, point, real_point):
        point=point.copy()
        real_point=real_point.copy()
        self.traj_x.append(point[0])
        self.traj_y.append(point[2])
        self.traj_z.append(-point[1])
        self.real_traj_x.append(real_point[1])
        self.real_traj_y.append(real_point[0])
        self.real_traj_z.append(0)

    def add_point_cloud(self, points):
        points=points.copy()
        for p in points:
            self.point_cloud.append(p)

    def add_err_point(self, point):
        self.err_x.append(point[0])
        self.err_y.append(point[1])
        self.err_angle.append(point[2])

    def draw_point_colored(self, points, color):
        #self.fig.add_trace(self.__pointsShape(points, "point cloud", "point cloud", self.blue))
        for idx, point in enumerate(points):
            self.fig.add_trace(self.__pointsShape1(point, "point cloud", "point cloud", "rgb({0}, {1}, {2})".format(color[idx][0],color[idx][1],color[idx][2])))
    

    def draw_point(self, points, color):
        self.fig.add_trace(self.__pointsShape(points, "point cloud", "point cloud", self.blue))
        
    def draw_image(self,img):
        self.video = img
    
    def __pointsShape1(self, point, name, text, color, visible=True, legend=True):
        return go.Scatter3d(
            x=[point[0]],
            y=[point[1]],
            z=[point[2]],
            mode='markers',
            marker=dict(size=2, color=color),
            visible=visible,
            text=text,
            name=name,
            hoverinfo='text',
            showlegend=legend
        )

    def __pointsShape(self, points, name, text, color, visible=True, legend=True):
        pos = pd.DataFrame(data=points, columns=['x', 'y', 'z'])
        return go.Scatter3d(
            x=pos['x'],
            y=pos['y'],
            z=pos['z'],
            mode='markers',
            marker=dict(size=2, color=color),
            visible=visible,
            text=text,
            name=name,
            hoverinfo='text',
            showlegend=legend
        )

    def __lineShape(self, points, name, text, color, visible=True, legend=True):
        pos = pd.DataFrame(data=points, columns=['x', 'y', 'z'])
        return go.Scatter3d(
            x=pos['x'],
            y=pos['y'],
            z=pos['z'],
            mode='lines',
            marker=dict(size=5, color=color),
            line=dict(
                color=color,
                width=5
            ),
            visible=visible,
            text=text,
            name=name,
            hoverinfo='text',
            showlegend=legend
        )
