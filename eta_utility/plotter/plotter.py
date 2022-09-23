import plotly.graph_objects as go
import dash
from dash import dcc
from dash import html
import plotly.io as pio
from dash.dependencies import Input, Output

from eta_utility import get_logger

log = get_logger("eta_utility.plotter")




class ETA_Plotter:
    """Plots multiple :class:`Linegraph` and :class:`Heatmap` figures in a single html file or displays them in a dashboard. Can also save multiple figures in seperate pdf and svg files.   
    
    :param subplots: Heatmap and linegraph figures created with :class:`Linegraph` and :class:`Heatmap`.
    """

    def __init__(self, *subplots):
        self.subplots = []
        for sub in subplots:
            self.subplots.append(sub)

    def dashboard(self, title='ETA-X Dashboard'):
        """Creates a dashboard with all figures and prints a link to the dashboard in the terminal.

        :param str title: Dashboard title. Defaults to 'ETA-X Dahsboard'.
        """
        external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
        app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

        container = [html.H1(title)]
        for sub in self.subplots:
            container.append(dcc.Graph(figure=sub.fig))

        app.layout = html.Div(container)

        if __name__ == '__main__':
            app.run_server(debug=True)

    def plot_html(self, filename='html_plot.html'):
        """Saves all figures in a single html file.

        :param str filename: Filepath including /filename.html. Without filepath the html file will be saved as 'html_plot.html' in the project directory.
        """

        with open(filename, 'w') as f:
            for sub in self.subplots:
                f.write(sub.fig.to_html(full_html=False, include_plotlyjs='cdn'))
        # print('html file created successfully.')

    def plot_pdf(self, filepath=''):
        """Saves all figures in seperate pdf files.

        :param str filepath: Directory for the pdf files. Without a directory the pdf files will be saved as consecutively numbered 'subplot.pdf' in the project directory. 
        """

        i = 0
        for sub in self.subplots:
            i += 1
            if filepath:
                path = filepath + "\subplot" + str(i) + ".pdf"
                pio.write_image(sub.fig, path, format='pdf')

            else:
                path = "subplot" + str(i) + ".pdf"
                pio.write_image(sub.fig, path, format='pdf')
        print('pdf plots created successfully.')

    def plot_svg(self, filepath=''):
        """Saves all figures in seperate svg files.

        :param str filepath: Directory for the svg files. Without a directory the svg files will be saved as consecutively numbered 'subplot.svg' in the project directory. 
        """

        i = 0
        for sub in self.subplots:
            i += 1
            if filepath:
                path = filepath + "\subplot" + str(i) + ".svg"
                pio.write_image(sub.fig, path, format='svg')

            else:
                path = "subplot" + str(i) + ".svg"
                pio.write_image(sub.fig, path, format='svg')
        print('svg plots created successfully.')

    def live_layout(self, update_interval, title):
        external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
        self.app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
        self.id = ''
        # globale id in Plotter class

        self.container = [html.H1(title),
                          dcc.Interval(
                              id='update interval',
                              disabled=False,  # if True, the counter will no longer update
                              interval=update_interval,  # increment the counter n_intervals every interval milliseconds
                              n_intervals=0,  # number of times the interval has passed
                              max_intervals=-1,  # number of times the interval will be fired.
                              # if -1, then the interval has no limit (the default)
                              # and if 0 then the interval stops running.
                          ),
                          ]
        self.app.layout = html.Div(self.container)

    def create_dcc_graph(self):
        dcc_graph = dcc.Graph(id=self.id)
        self.container.append(dcc_graph)

    def create_callback(self, sub):
        self.create_dcc_graph()

        @self.app.callback(
            Output(self.id, 'figure'),
            Input('update interval', 'n_intervals')
        )
        def update_Graph(n_intervals):
            return sub.live()

    def update_id(self, sub):
        self.id = str(id(sub))

    def id_update_and_callback_for_subplots(self):
        for sub in self.subplots:
            self.update_id(sub)
            self.create_callback(sub)

    def play(self):
        if __name__ == '__main__':
            self.app.run_server(debug=True)

    def plot_live(self, update_interval=1, title='ETA-X Live'):
        self.live_layout(update_interval, title)
        self.id_update_and_callback_for_subplots()
        self.play()


class Linegraph():
    """Creates a simple Linegraph as a plotly Figure. 
    
    :param x: Pandas dataframe column or array-like object for the x-coordinates. 
    :param str title: Title of the Linegraph.
    :param str xaxis_title: X-axis title.
    :param str yaxis_title: Y-axis title.
    :param int height: Height of the figure. 0 for autosize.
    :param int width: Width of the figure. 0 for autosize.
    """

    def __init__(self, x, title='', xaxis_title='', yaxis_title='', height=0, width=0):
        self.x = x
        self.title = title
        self.xaxis_title = xaxis_title
        self.yaxis_title = yaxis_title
        self.height = height
        self.width = width
        self.fig = go.Figure()
        self._layout_()
        self.graph_data = [] #für live plotter


    def _layout_(self):
        """Layout settings for :class:'Linegraph'.
        """

        margin_l = 150
        margin_r = 200

        if self.height and self.width:
            self.fig.update_layout(title=self.title,
                                   xaxis_title=self.xaxis_title,
                                   yaxis_title=self.yaxis_title,
                                   height=self.height,
                                   width=self.width,
                                   margin=dict(l=margin_l, r=margin_r),
                                   plot_bgcolor='rgb(255,255,255)'
                                   )


        elif self.height or self.width:
            if self.height:
                self.fig.update_layout(title=self.title,
                                       xaxis_title=self.xaxis_title,
                                       yaxis_title=self.yaxis_title,
                                       height=self.height,
                                       margin=dict(l=margin_l, r=margin_r),
                                       plot_bgcolor='rgb(255,255,255)'
                                       )
            else:
                self.fig.update_layout(title=self.title,
                                       xaxis_title=self.xaxis_title,
                                       yaxis_title=self.yaxis_title,
                                       width=self.width,
                                       margin=dict(l=margin_l, r=margin_r),
                                       plot_bgcolor='rgb(255,255,255)'
                                       )

        else:
            self.fig.update_layout(title=self.title,
                                   xaxis_title=self.xaxis_title,
                                   yaxis_title=self.yaxis_title,
                                   margin=dict(l=margin_l, r=margin_r),
                                   plot_bgcolor='rgb(255,255,255)'
                                   )
        
        self.fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True, gridcolor='lightgrey', zeroline=False)
        self.fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True, gridcolor='lightgrey', zeroline=False)


    def line(self, y, name='', dash='solid', width=2, fill='none', color=''):
        """Adds a line to the linegraph.
        
        :param y: Pandas dataframe column or array-like object for the y-coordinates. 
        :param str name: Name of the line to be displayed in the legend. Defaults to '' for the name of the pandas data frame column. 
        :param str dash: Line type - one of the following dash styles: ['solid', 'dot', 'dash', 'longdash', 'dashdot', 'longdashdot']. Defaults to 'solid'.
        :param int width: Line width. Defaults to 2.
        :param str fill: Area that needs to be filled between this line and one of the following values: ['none', 'tozeroy', 'tozerox', 'tonexty', 'tonextx', 'toself', 'tonext']. Defaults to 'none'.
        :param str color: Color of the line. May be specified as:

            - A hex string (e.g. '#ff0000')
            - An rgb / rgba string (e.g. 'rgb(255,0,0)')
            - An hsl / hsla string (e.g. 'hsl(0,100%,50%)')
            - An hsv / hsva string (e.g. 'hsv(0,100%,100%)')
            - A named CSS color:

                aliceblue, antiquewhite, aqua, aquamarine, azure, beige, bisque, black, blanchedalmond, blue, blueviolet, brown, burlywood, cadetblue,
                chartreuse, chocolate, coral, cornflowerblue, cornsilk, crimson, cyan, darkblue, darkcyan, darkgoldenrod, darkgray, darkgrey, darkgreen, darkkhaki, darkmagenta, darkolivegreen, darkorange,
                darkorchid, darkred, darksalmon, darkseagreen, darkslateblue, darkslategray, darkslategrey, darkturquoise, darkviolet, deeppink, deepskyblue,
                dimgray, dimgrey, dodgerblue, firebrick, floralwhite, forestgreen, fuchsia, gainsboro, ghostwhite, gold, goldenrod, gray, grey, green,
                greenyellow, honeydew, hotpink, indianred, indigo, ivory, khaki, lavender, lavenderblush, lawngreen, lemonchiffon, lightblue, lightcoral, lightcyan,
                lightgoldenrodyellow, lightgray, lightgrey, lightgreen, lightpink, lightsalmon, lightseagreen, lightskyblue, lightslategray, lightslategrey,
                lightsteelblue, lightyellow, lime, limegreen, linen, magenta, maroon, mediumaquamarine, mediumblue, mediumorchid, mediumpurple,
                mediumseagreen, mediumslateblue, mediumspringgreen, mediumturquoise, mediumvioletred, midnightblue, mintcream, mistyrose, moccasin, navajowhite, navy,
                oldlace, olive, olivedrab, orange, orangered, orchid, palegoldenrod, palegreen, paleturquoise, palevioletred, papayawhip, peachpuff, peru, pink,
                plum, powderblue, purple, red, rosybrown, royalblue, rebeccapurple, saddlebrown, salmon, sandybrown, seagreen, seashell, sienna, silver,
                skyblue, slateblue, slategray, slategrey, snow, springgreen, steelblue, tan, teal, thistle, tomato, turquoise, violet, wheat, white, whitesmoke,
                yellow, yellowgreen
        """
        name = name if name else y.name

        if color:
            self.fig.add_trace(go.Scatter(x=self.x,
                                          y=y,
                                          name=name,
                                          line=dict(dash=dash,
                                                    width=width,
                                                    color=color,
                                                    ),
                                          fill=fill,
                                          )
                               )

            # für live plotter
            line_data = go.Scatter(x=self.x,
                                   y=y,
                                   name=name,
                                   line=dict(dash=dash,
                                             width=width,
                                             color=color,
                                             ),
                                   fill=fill,
                                   )

            self.graph_data.append(line_data)

        else:
            self.fig.add_trace(go.Scatter(x=self.x,
                                          y=y,
                                          name=name,
                                          line=dict(dash=dash,
                                                    width=width,
                                                    ),
                                          fill=fill,
                                          )
                               )

            # für live plotter
            line_data = go.Scatter(x=self.x,
                                   y=y,
                                   name=name,
                                   line=dict(dash=dash,
                                             width=width,
                                             ),
                                   fill=fill,
                                   )

            self.graph_data.append(line_data)

    def live(self, arg):
        return arg

        # f_live = Linegraph(live_dates, 'Test', 'Zeit', 'y-Achse')
        # f_live.line(live_df["s_price_electricity"])

        # self.fig = go.Figure(data=self.graph_data)
        # self._layout_()
        # return self.fig

        # return f_live.fig

    def show(self):
        """Displays the Linegraph in the Browser for observation."""

        self.fig.show()

    def plot_html(self, filename='Linegraph.html'):
        """Saves the Linegraph as an html file.
        
        :param str filename: Filepath including /filename.html. Without filepath the html file will be saved as 'Linegraph.html' in the project directory.
        """

        with open(filename, 'w') as f:
            f.write(self.fig.to_html(full_html=False, include_plotlyjs='cdn'))
        print('Linegraph html created successfully.')

    def plot_pdf(self, filename='Linegraph.pdf'):
        """Saves the Linegraph as a pdf file.
        
        :param str filename: Filepath including /filename.pdf. Without filepath the pdf file will be saved as 'Linegraph.pdf' in the project directory.
        """
        pio.write_image(self.fig, filename, format='pdf')
        print('Linegraph pdf created successfully.')

    def plot_svg(self, filename='Linegraph.svg'):
        """Saves the Linegraph as an svg file.
        
        :param str filename: Filepath including /filename.svg. Without filepath the svg file will be saved as 'Linegraph.svg' in the project directory.
        """
        pio.write_image(self.fig, filename, format='svg')
        print('Linegraph svg created successfully.')


class Heatmap:
    """Creates a simple Heatmap as a plotly Figure. 
    
    :param x: Pandas dataframe column or array-like object for the x-coordinates.
    :param str title: Title of the Heatmap.
    :param str xaxis_title: X-axis title.
    :param int height: Height of the figure. 0 for autosize.
    :param int width: Width of the figure. 0 for autosize.
    :param str colorscale: Colorscale for the Heatmap which may be specified as:

        - A list of 2-element lists where the first element is the normalized color level value (starting at 0 and ending at 1), and the second item is a valid color string. (e.g. [[0, 'green'], [0.5, 'red'], [1.0, 'rgb(0, 0, 255)']])
        - One of the following named colorscales:

            ['aggrnyl', 'agsunset', 'algae', 'amp', 'armyrose', 'balance',
            'blackbody', 'bluered', 'blues', 'blugrn', 'bluyl', 'brbg',
            'brwnyl', 'bugn', 'bupu', 'burg', 'burgyl', 'cividis', 'curl',
            'darkmint', 'deep', 'delta', 'dense', 'earth', 'edge', 'electric',
            'emrld', 'fall', 'geyser', 'gnbu', 'gray', 'greens', 'greys',
            'haline', 'hot', 'hsv', 'ice', 'icefire', 'inferno', 'jet',
            'magenta', 'magma', 'matter', 'mint', 'mrybm', 'mygbm', 'oranges',
            'orrd', 'oryel', 'peach', 'phase', 'picnic', 'pinkyl', 'piyg',
            'plasma', 'plotly3', 'portland', 'prgn', 'pubu', 'pubugn', 'puor',
            'purd', 'purp', 'purples', 'purpor', 'rainbow', 'rdbu', 'rdgy',
            'rdpu', 'rdylbu', 'rdylgn', 'redor', 'reds', 'solar', 'spectral',
            'speed', 'sunset', 'sunsetdark', 'teal', 'tealgrn', 'tealrose',
            'tempo', 'temps', 'thermal', 'tropic', 'turbid', 'twilight',
            'viridis', 'ylgn', 'ylgnbu', 'ylorbr', 'ylorrd'].

            Appending '_r' to a named colorscale reverses it.
    """

    def __init__(self, x, title='', xaxis_title='', height=0, width=0, colorscale='Greys'):
        self.x = x
        self.title = title
        self.xaxis_title = xaxis_title
        self.height = height
        self.width = width
        self.colorscale = colorscale
        self.y = []
        self.z = []
        self.fig = go.Figure(data=go.Heatmap(x=self.x, y=self.y, z=self.z, colorscale=self.colorscale))
        self._layout_()

    def line(self, z, name=''):
        """Adds a line of blocks to the Heatmap.

        :param z: Pandas dataframe column or array-like object to be displayed in this line.
        :param str name: Name of the line of blocks to be displayed on the y-axis. Defaults to '' for the name of the pandas dataframe column.

        """
        name = name if name else z.name

        self.y.append(name)
        self.z.append(z)
        self.fig = go.Figure(data=go.Heatmap(x=self.x, y=self.y, z=self.z, colorscale=self.colorscale))
        self._layout_()

    def _layout_(self):
        """Layout settings for :class:'Heatmap'.
        """
        
        margin_l = 150
        margin_r = 200

        if self.height and self.width:
            self.fig.update_layout(title=self.title,
                                   xaxis_title=self.xaxis_title,
                                   height=self.height,
                                   width=self.width,
                                   margin=dict(l=margin_l, r=margin_r),
                                   yaxis_tickangle=-30
                                   )

        elif self.height or self.width:
            if self.height:
                self.fig.update_layout(title=self.title,
                                       xaxis_title=self.xaxis_title,
                                       height=self.height,
                                       margin=dict(l=margin_l, r=margin_r),
                                       yaxis_tickangle=-30
                                       )
            else:
                self.fig.update_layout(title=self.title,
                                       xaxis_title=self.xaxis_title,
                                       width=self.width,
                                       margin=dict(l=margin_l, r=margin_r),
                                       yaxis_tickangle=-30
                                       )
        else:
            self.fig.update_layout(title=self.title,
                                   xaxis_title=self.xaxis_title,
                                   margin=dict(l=margin_l, r=margin_r),
                                   # paper_bgcolor="LightSteelBlue",
                                   yaxis_tickangle=-30,
                                   )

        self.fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
        self.fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)


    def show(self):
        """Displays the Heatmap in the Browser for observation."""

        self.fig.show()

    def plot_html(self, filename='Heatmap.html'):
        """Saves the Heatmap as an html file.
        
        :param str filename: Filepath including /filename.html. Without filepath the html file will be saved as 'Heatmap.html' in the project directory.
        """

        with open(filename, 'w') as f:
            f.write(self.fig.to_html(full_html=False, include_plotlyjs='cdn'))
        print('Heatmap html created successfully.')

    def plot_pdf(self, filename='Heatmap.pdf'):
        """Saves the Heatmap as a pdf file.
        
        :param str filename: Filepath including /filename.pdf. Without filepath the pdf file will be saved as 'Heatmap.pdf' in the project directory.
        """

        pio.write_image(self.fig, filename, format='pdf')
        print('Heatmap pdf created successfully.')

    def plot_svg(self, filename='Heatmap.svg'):
        """Saves the Heatmap as an svg file.
        
        :param str filename: Filepath including /filename.svg. Without filepath the svg file will be saved as 'Heatmap.svg' in the project directory.
        """
        
        pio.write_image(self.fig, filename, format='svg')
        print('Heatmap svg created successfully.')

