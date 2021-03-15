import os 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import SPP_v5

from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.properties import ObjectProperty, StringProperty
from kivy.core.window import Window
from kivymd.uix.filemanager import MDFileManager
from kivymd.uix.datatables import MDDataTable
from kivy.uix.widget import Widget
from kivymd.uix.button import MDRaisedButton
from kivy.metrics import dp
from kivymd.app import MDApp
from kivymd.uix.dialog import MDDialog
from kivymd.uix.list import OneLineIconListItem
from kivymd.uix.button import MDFlatButton
from kivymd.uix.screen import MDScreen
from kivy.garden.mapview import MapView
from kivy.garden.mapview import MapMarker

import matplotlib
matplotlib.use("module://kivy.garden.matplotlib.backend_kivy")
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg # lub FigureCanvas
from matplotlib.pyplot import rc
import matplotlib.patches as mpatches


KV = ''' 

<Item>
    IconLeftWidget:
        icon: root.icon
        
<MapContainer>:
    GridLayout:
        cols:1
        size_hint: 1, 0.89
        
        BoxLayout:
            
            Map:
                id: map
                lat: 52.20
                lon: 21.00
                zoom: 10
      

<DataTables>
    # list_data: data_tab
        
    MDBoxLayout:
        orientation: "vertical"
        adaptive_height: True

        MDBottomNavigation:
            
            MDBottomNavigationItem:
                text: 'Show table'
                icon: 'database'
                on_tab_release: root.open_table()
    
            MDBottomNavigationItem:
                text: 'Save table'
                icon: 'database-export'
                on_tab_release: root.export_dialog()
                

<Plots>:
    plot_1: plot1
    plot_2: plot2
    plot_3: plot3
    plot_4: plot4

    GridLayout:
        cols:2
        size_hint: 1, 0.89
        
        BoxLayout:
            orientation: "vertical"
            id: plot1
            
        BoxLayout:
            orientation: "vertical"
            id: plot2

                
        BoxLayout:
            orientation: "vertical"
            id: plot3

        
        BoxLayout:
            orientation: "vertical"
            id: plot4



<Content>
    id: cont
    orientation: "vertical"
    spacing: "12dp"
    size_hint_y: None
    height: "120dp"

    MDTextField:
        id: abs_path
        hint_text: "Path to the directory"

    MDTextField:
        id: f_name
        hint_text: "File Name"
        
        
        
<TooltipMDFloatingActionButton@MDFloatingActionButton+MDTooltip>

<Check@MDCheckbox>:
    group: 'group'
    size_hint: None, None
    size: dp(48), dp(48)
        
<Check_weights@MDCheckbox>:
    group: 'group_weights'
    size_hint: None, None
    size: dp(48), dp(48)
    
<ContentNavigationDrawer>:
    orientation: "vertical"
    padding: "8dp"
    spacing: "8dp"
    
    AnchorLayout:
        anchor_x: "left"
        size_hint_y: None
        height: avatar.height

        Image:
            id: avatar
            size_hint: None, None
            size: "300dp", "130dp"
            source: "WGiK-znak.png"

    ScrollView:

        MDList:
            
            OneLineAvatarListItem:
                text: "Main Window"
                on_release:
                    root.nav_drawer.set_state("close")
                    root.screen_manager.current = "scr 1"
            
                IconLeftWidget:
                    icon: "satellite-variant"

            OneLineAvatarListItem:
                text: "Positions | DOP"
                on_release:
                    root.nav_drawer.set_state("close")
                    root.screen_manager.current = "scr 2"
                    
                IconLeftWidget:
                    icon: "database"
                    
            OneLineAvatarListItem:
                text: "Plots and Charts"
                on_release:
                    root.nav_drawer.set_state("close")
                    root.screen_manager.current = "scr 3"
                    
                IconLeftWidget:
                    icon: "chart-bell-curve-cumulative"

            OneLineAvatarListItem:
                text: "Map"
                on_release:
                    root.nav_drawer.set_state("close")
                    root.screen_manager.current = "scr 4"
                    
                IconLeftWidget:
                    icon: "google-maps"

Screen:
         
    MDToolbar:
        id: toolbar
        pos_hint: {"top": 1}
        elevation: 10
        title: "Single Point Positioning"
        left_action_items: [["menu", lambda x: nav_drawer.set_state("open")]]
    


                
    MDNavigationLayout:
        x: toolbar.height
        
                
        ScreenManager:
            id: screen_manager

            Screen:
                name: "scr 1"
                
                    
                FloatLayout:
                    

                    MDRoundFlatIconButton:
                        id: upload_files
                        text: "Upload files"
                        icon: "file-upload-outline"
                        pos_hint: {'center_x': .12, 'center_y': .82}
                        on_release: app.file_manager_open()
                        
                    MDTextField:
                        id: nav_file
                        pos_hint: {'center_x': .33, 'center_y': .81}
                        size_hint: (.15, None)
                        hint_text: "Navigation file"
                        text: ""
                        readonly: True 
                        
                    MDTextField:
                        id: obs_file
                        pos_hint: {'center_x': .52, 'center_y': .81}
                        size_hint: (.2, None)
                        hint_text: "Observation file"
                        text: ""
                        readonly: True 
                    
                    MDLabel:
                        pos_hint: {'center_x': .53, 'center_y': .7}
                        text: "Select observations:"   
                        
                    MDLabel:
                        id: code_label
                        pos_hint: {'center_x': .78, 'center_y': .7}
                        text: "Code" 
                        
                    MDLabel:
                        id: smoothed_code_label
                        pos_hint: {'center_x': .93, 'center_y': .7}
                        text: "Smoothed code" 
                        
                    MDLabel:
                        # id: smoothing_window
                        pos_hint: {'center_x':1.095, 'center_y': .7}
                        text: "|    *Smoothing window:" 
                        
                    MDTextField:
                        id: smoothing_window
                        pos_hint: {'center_x': 0.85, 'center_y': .7}
                        size_hint: (.07, None)
                        readonly: True
                        line_color_normal: [255, 0, 0, 1]
                        helper_text: "[seconds]"
                        helper_text_mode: "persistent"
                        
                    MDLabel:
                        pos_hint: {'center_x': .53, 'center_y': .63}
                        text: "Use weight matrix:" 
                        
                    MDLabel:
                        id: weight_matrix_yes_label
                        pos_hint: {'center_x': .78, 'center_y': .63}
                        text: "Yes" 
                        
                    MDLabel:
                        id: weight_matrix_no_label
                        pos_hint: {'center_x': .93, 'center_y': .63}
                        text: "No"     
                        
                    MDLabel:
                        pos_hint: {'center_x': .53, 'center_y': .56}
                        text: "Elevation mask:" 
                        
                    MDTextField:
                        id: el_mask
                        pos_hint: {'center_x': .285, 'center_y': .56}
                        size_hint: (.09, None)
                        helper_text: "[degrees]"
                        helper_text_mode: "persistent"
                        
                    MDLabel:
                        pos_hint: {'center_x': .53, 'center_y': .49}
                        text: "Epochs number:" 
                        
                    MDTextField:
                        id: epochs_number
                        pos_hint: {'center_x': .285, 'center_y': .49}
                        size_hint: (.09, None)
                        
                        
                    FloatLayout:
    
                        Check:
                            id: code_check
                            pos_hint: {'center_x': .25, 'center_y': .7}
                            on_active: app.on_checkbox_active(*args)
                    
                        Check:
                            id: smoothed_code_check
                            pos_hint: {'center_x': .4, 'center_y': .7}    
                            on_active: app.on_checkbox_active(*args)
                            
                        Check_weights:
                            id: weight_matrix_yes_check
                            pos_hint: {'center_x': .25, 'center_y': .63}
                            on_active: app.on_checkbox_active_weights(*args)
                    
                        Check_weights:
                            id: weight_matrix_no_check
                            pos_hint: {'center_x': .4, 'center_y': .63}    
                            on_active: app.on_checkbox_active_weights(*args)
                        
                        
                    FloatLayout:        
                        MDRoundFlatIconButton:
                            id: calc_pos
                            text: "Calculate positions"
                            icon: "math-compass"
                            pos_hint: {'center_x': .48, 'center_y': .4}
                            width: dp(170)
                            on_release:
                                app.get_data()

                        MDSpinner:
                            id: spinner
                            size_hint: None, None
                            size: dp(46), dp(46)
                            pos_hint: {'center_x': .5, 'center_y': .5}
                            active: False
                            
                            
                    
                    FloatLayout:
                        
                        TooltipMDFloatingActionButton: #go to tables with pos and dop
                            id: btn1
                            icon: "blank"
                            elevation: 0
                            md_bg_color: [1,1,1,0]
                            pos_hint: {"center_x": .2, "center_y": 0.1}
                            on_release:screen_manager.current = "scr 2"
                            
                           
                        TooltipMDFloatingActionButton: # go to plots
                            id: btn2
                            icon: "blank"
                            elevation: 0
                            md_bg_color: [1,1,1,0]
                            pos_hint: {"center_x": .5, "center_y": 0.1}
                            on_release: 
                                screen_manager.current = "scr 3"

                        TooltipMDFloatingActionButton: # go to map
                            id: btn3
                            icon: "blank"
                            elevation: 0
                            md_bg_color: [1,1,1,0]
                            pos_hint: {"center_x": .8, "center_y": 0.1}
                            on_release: 
                                screen_manager.current = "scr 4"
                            
                            
                            

                    
            
                            
            Screen:
                name: "scr 2"
                
                DataTables:
                    id: data_tablee
                                                  
            Screen:
                name: "scr 3"
                
                Plots:
                    id: prob
                    
            Screen:
                name: "scr 4"
                
                MapContainer:
                    id: map
                
        MDNavigationDrawer:
            id: nav_drawer
        
            ContentNavigationDrawer:
                screen_manager: screen_manager
                nav_drawer: nav_drawer              

        
'''

        
class DataTables(MDScreen):
    dialog = None
    data = ObjectProperty()
    
    def open_table(self):
        data_row = []
        self.headers = ["Epoch", "X", "Y", "Z", "GDOP", "PDOP", "TDOP", "HDOP", "VDOP"]

        for i in range(len(self.data["Epoch"])):
            x = tuple([round(self.data[h][i], 3) for h in self.headers])
            data_row.append(x)
            
        data_tables = MDDataTable(
            size_hint=(0.9, 0.65),
            pos_hint={"center_x": 0.5, "center_y": 0.5},
            rows_num=len(self.data["Epoch"]),
            column_data = [
                ("Epoch [SOD]", dp(25)),
                ("X [m]", dp(25)),
                ("Y [m]", dp(25)),
                ("Z [m]", dp(25)),
                ("GDOP", dp(25)),
                ("PDOP", dp(25)),
                ("TDOP", dp(25)),
                ("HDOP", dp(25)),
                ("VDOP", dp(25)),
                ],
                
            row_data = data_row
            )
        
        data_tables.ids.container.add_widget(
            Widget(size_hint_y=None, height="5dp")
        )
        data_tables.ids.container.add_widget(
            MDRaisedButton(
                text="CLOSE",
                pos_hint={"right": 1},
                on_release=lambda x: self.remove_widget(data_tables),
            )
        )
        
        self.add_widget(data_tables)
        
        
    def export_dialog(self):

        if not self.dialog:
            self.dialog = MDDialog(
                title="Save File",
                type="custom",
                content_cls=Content(),
                buttons=[
                    MDFlatButton(
                        text="CANCEL", 
                        on_release=self.close_dialog
                    ),
                    MDFlatButton(
                        text="OK",
                        on_release=self.export_table
                    ),
                ],
            )
        self.dialog.set_normal_height()
        self.dialog.open()
        
    def close_dialog(self, inst):
        self.dialog.dismiss()
        
    def export_table(self, inst):
        
        path = self.dialog.content_cls.ids.abs_path.text
        file_name = self.dialog.content_cls.ids.f_name.text
        file_name = file_name + ".xlsx" 
        path = os.path.join(path, file_name)
       
        df = pd.DataFrame(self.data, columns = self.headers)
        df.to_excel(path, index = False, header=True)
        self.dialog.dismiss()
  

        
class Plots(BoxLayout):
    pass

class Item(OneLineIconListItem):
    divider = None
    icon = StringProperty()
    
class LocationPopupMenu(MDDialog):
      
    def __init__(self, coors):
        super().__init__()
        
        deg_sign = "\u00b0" 
        B = SPP_v5.decimalDeg2dms(coors[0])
        B = f"{B[0]}{deg_sign} {B[1]}' {round(B[2])}'' "      
        L = SPP_v5.decimalDeg2dms(coors[1])
        L = f"{L[0]}{deg_sign} {L[1]}' {round(L[2])}''"  
        H = f"{round(coors[2], 3)} m" 
        
        self.dialog = MDDialog(
            title="INFO",
            type='simple',
            size_hint=(0.5, 0.5),
            items=[Item(text=B, icon="alpha-b"),
                   Item(text=L, icon="alpha-l"),
                   Item(text=H, icon="alpha-h")],
            )

        self.dialog.open()

      
class LocationMarker(MapMarker):
    coors = []
        
    def on_release(self):
        menu = LocationPopupMenu(self.coors)


class MapContainer(BoxLayout):
    pass


class Map(MapView):
    pass


class Content(BoxLayout):
    pass
        

class ContentNavigationDrawer(BoxLayout):
    screen_manager = ObjectProperty()
    nav_drawer = ObjectProperty()


class SPPApp(MDApp):  
    plot = ObjectProperty()
    dialog = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        Window.bind(on_keyboard=self.events)
        self.manager_open = False
        self.file_manager = MDFileManager(
            exit_manager=self.exit_manager,
            select_path=self.select_path
        )
        self.file_manager.ext = [".txt", ".19n", ".asc", ".15n"]
        self.nav_path = None
        self.obs_path = None
        self.obs_type = None
        self.weight_matrix = True

    def build(self):
        return Builder.load_string(KV)
    
    def file_manager_open(self):
        self.file_manager.show('/')  # output manager to the screen
        self.manager_open = True

    def select_path(self, path):
        '''It will be called when you click on the file name
        or the catalog selection button.

        :type path: str;
        :param path: path to the selected directory or file;
        '''
        
        file = path.split('/')
        
        if file[-1][-1] == "n":
            self.nav_path = path
            self.root.ids.nav_file.text = str(file[-1])
        else:
            self.obs_path = path
            self.root.ids.obs_file.text = str(file[-1])
        
        self.exit_manager()


    def exit_manager(self, *args):
        '''Called when the user reaches the root of the directory tree.'''

        self.manager_open = False
        self.file_manager.close()

    def events(self, instance, keyboard, keycode, text, modifiers):
        '''Called when buttons are pressed on the mobile device.'''

        if keyboard in (1001, 27):
            if self.manager_open:
                self.file_manager.back()
        return True
    
    def on_checkbox_active(self, checkbox, value):
        
        if self.root.ids.code_check == checkbox: # code observations
            self.obs_type = "raw"
            self.root.ids.code_label.bold = True
            self.root.ids.smoothed_code_label.bold = False
            self.root.ids.smoothing_window.readonly = True
            self.root.ids.smoothing_window.line_color_normal = [255, 0, 0, 1]
            self.smooth_window = 1
            
        elif self.root.ids.smoothed_code_check == checkbox: # smoothed code observations
            self.obs_type = "smooth"
            self.root.ids.smoothed_code_label.bold = True
            self.root.ids.code_label.bold = False
            self.root.ids.smoothing_window.readonly = False
            self.root.ids.smoothing_window.line_color_normal = [0, 255, 0, 1]
            self.smooth_window = self.root.ids.smoothing_window
        
            
    def on_checkbox_active_weights(self, checkboxx, valuee):
        
        if self.root.ids.weight_matrix_yes_check == checkboxx: 
            self.weight_matrix = True
            self.root.ids.weight_matrix_yes_label.bold = True
            self.root.ids.weight_matrix_no_label.bold = False
        
        elif self.root.ids.weight_matrix_no_check == checkboxx:
            self.weight_matrix = False
            self.root.ids.weight_matrix_no_label.bold = True
            self.root.ids.weight_matrix_yes_label.bold = False
            
    def get_data(self):
        
        # if (self.obs_path or self.nav_path or self.obs_type) == None:
        if self.obs_path == None or self.nav_path == None or self.obs_type == None:
            if self.obs_path == None:
                title = "Upload observation file!"
            
            elif self.nav_path == None:
                title = "Upload navigation file!"
            
            elif self.obs_type == None:
                title = "Choose observations!"
                
            self.dialog = MDDialog(
                title=title,
                buttons=[MDFlatButton(
                        text="OK", text_color=self.theme_cls.primary_color,
                        on_release=self.close_dialog)])
            self.dialog.set_normal_height()
            self.dialog.open()
            
        else:
            self.get_positions()
            self.show_buttons()
        
    def close_dialog(self, inst):
        self.dialog.dismiss()
        
    def get_positions(self): 
                
        if self.obs_type == "smooth":
            if self.smooth_window.text == "":
                self.smooth_window = 60
            
            else:
                self.smooth_window = int(self.smooth_window.text)
                
        
        if self.root.ids.epochs_number.text == "":
            self.number_epochs = 100
        
        else:
            self.number_epochs = int(self.root.ids.epochs_number.text)
        
        
        if self.root.ids.el_mask.text == "":
            self.el_mask = 10
        
        else:
            self.el_mask = int(self.root.ids.el_mask.text)
            
        # print(self.obs_type, self.smooth_window, self.number_epochs, self.weight_matrix, self.el_mask)
        
        self.pos, start_epoch = SPP_v5.main(self.obs_path, self.nav_path, self.obs_type, 
                                    self.smooth_window, self.number_epochs,
                                    self.weight_matrix, self.el_mask)
        
        
        
        self.headers = ["Epoch", "X", "Y", "Z", "GDOP", "PDOP", "TDOP", "HDOP", "VDOP", "m0", "V_hat"]      
        self.data = dict()
        
        for head in self.headers:
            self.data[head] = []
            
        for epoch, values in self.pos.items():
            self.data["Epoch"].append(epoch+start_epoch)
            self.data["X"].append(values[0][0])
            self.data["Y"].append(values[0][1])
            self.data["Z"].append(values[0][2])
            self.data["GDOP"].append(values[1][0])
            self.data["PDOP"].append(values[1][1])
            self.data["TDOP"].append(values[1][2])
            self.data["HDOP"].append(values[1][3])
            self.data["VDOP"].append(values[1][4])
            self.data["m0"].append(values[2])
            self.data["V_hat"].append(values[3])
            
            
        B, L, H = SPP_v5.hirvonen(np.mean(self.data["X"]), np.mean(self.data["Y"]), np.mean(self.data["Z"])) 
        self.add_markers([B, L, H])
        
        self.root.ids.data_tablee.data = self.data
        self.plots()
    
    
    def add_markers(self, coors):
        
        marker = LocationMarker(lat=coors[0], lon=coors[1], source=r"C:\Users\Bartek\Desktop\STUDIA\INZYNIERKA\App\marker.png")
        marker.coors = coors
        self.root.ids.map.ids.map.add_marker(marker)
        

    def show_buttons(self):
        
        self.root.ids.btn1.elevation = 10
        self.root.ids.btn1.icon = "database"
        self.root.ids.btn1.md_bg_color = self.theme_cls.primary_color
        self.root.ids.btn1.tooltip_text = "Table with positions and DOP factors"
        
        self.root.ids.btn2.elevation = 10
        self.root.ids.btn2.icon = "chart-bell-curve-cumulative"
        self.root.ids.btn2.md_bg_color = self.theme_cls.primary_color
        self.root.ids.btn2.tooltip_text = "Plots and Charts"
        
        self.root.ids.btn3.elevation = 10
        self.root.ids.btn3.icon = "google-maps"
        self.root.ids.btn3.md_bg_color = self.theme_cls.primary_color
        self.root.ids.btn3.tooltip_text = "Positions on the map"
    
    def plots(self):
        self.dop_plot()
        self.skyplot()
        self.m0_plot()
        self.V_hat_plot()
        
    def dop_plot(self):
        
        dop_data = {key: self.data[key] for key in self.headers[4:]}
        self.nb_epoch = self.data["Epoch"]
        
        fig = plt.figure() #utworzenie wykresy
        ax1 = fig.add_subplot(111) #dodanie wykresu
        ax1.plot(self.nb_epoch, dop_data["GDOP"], label="GDOP")
        ax1.plot(self.nb_epoch, dop_data["PDOP"], label="PDOP")
        ax1.plot(self.nb_epoch, dop_data["TDOP"], label="TDOP")
        ax1.plot(self.nb_epoch, dop_data["HDOP"], label="HDOP")
        ax1.plot(self.nb_epoch, dop_data["VDOP"], label="VDOP")
        
        leg = ax1.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
                         ncol=5, mode="expand", borderaxespad=0.)
        
        for line, text in zip(leg.get_lines(), leg.get_texts()):
            text.set_color(line.get_color())
        
        for item in leg.legendHandles:
            item.set_visible(False)
            
        ax1.grid()
        
        cnv = FigureCanvasKivyAgg(fig) 
        self.root.ids.prob.ids.plot1.add_widget(cnv)
        cnv.draw()
    
    def m0_plot(self):
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        if self.obs_type == "raw":
            ax.plot(self.nb_epoch, self.data["m0"], color="green", label=r"$m_{0_{raw}}$")
        
        else:
            ax.plot(self.nb_epoch, self.data["m0"], color="red", label=r"$m_{0_{smoothed}}$")
            
        ax.legend(loc='upper right')
        ax.grid()
        
        cnv = FigureCanvasKivyAgg(fig) 
        self.root.ids.prob.ids.plot3.add_widget(cnv)
        cnv.draw()
        
    def V_hat_plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        if self.obs_type == "raw":
            ax.plot(self.nb_epoch, self.data["V_hat"], color="green", label=r"$\hat v_{raw}$")
        
        else:
            ax.plot(self.nb_epoch, self.data["V_hat"], color="red", label=r"$\hat v_{smoothed}$")
            
        ax.legend(loc='upper left')
        ax.grid()
        
        cnv = FigureCanvasKivyAgg(fig) 
        self.root.ids.prob.ids.plot4.add_widget(cnv)
        cnv.draw()
        
    
    def skyplot(self):
        
        data = {key: self.pos[key][-1] for key in self.pos.keys()}
        # print(data)
        
        rc('grid', color='gray', linewidth=1, linestyle='--')
        fontsize = 10
        rc('xtick', labelsize = fontsize)
        rc('ytick', labelsize = fontsize)
        rc('font', size = fontsize)
        
        # colors
        green   ='#467821'
        
        fig = plt.figure()
        plt.subplots_adjust(bottom= 0.08, 
                            top   = 0.92,
                            left  = 0.005, 
                            right = 0.74)
        
        ax = fig.add_subplot(111, polar=True) # define a polar type of coordinates
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        
        for (sv, el, az) in data[0]: 
             ax.annotate(str(sv),                            # show sat number
                        xy=(np.radians(az), 90-el),          # theta, radius # satellite position
                        bbox=dict(boxstyle="round", fc = green, alpha = 0.5),
                        horizontalalignment='center',
                        verticalalignment='center')
             
        gps = mpatches.Patch(color=green,  label='{:02.0f} GPS'.format(len(data[0])))
        plt.legend(handles=[gps], bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

        # axis ticks descriptions    
        ax.set_yticks(range(0, 90+10, 10))                   # Define the yticks
        yLabel = ['90', '', '', '60', '', '', '30', '', '', '']
        ax.set_yticklabels(yLabel)
        
        cnv = FigureCanvasKivyAgg(fig) 
        self.root.ids.prob.ids.plot2.add_widget(cnv)
        cnv.draw()
        
        
        
if __name__ == "__main__":
    SPPApp().run()

        
    
    