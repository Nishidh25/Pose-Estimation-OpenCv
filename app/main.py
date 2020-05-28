# -*- coding: utf-8 -*-
"""
Created on Thu May 28 21:43:59 2020

@author: Nishidh Shekhawat
"""
import kivy
from kivy.app import App
from kivy.uix.camera import Camera
from kivy.core.window import Window
from kivy.uix.boxlayout import BoxLayout

class MainApp(App):
    
    def build(self):
        Window.bind(on_request_close=self.on_request_close)
        main_layout = BoxLayout(orientation="vertical")
        cam = self.camera()
        cam.play = True
        main_layout.add_widget(cam)
        return main_layout
        
    def camera(self):
        return Camera(index=0, resolution=(640,480))
    
    def on_request_close(self, *args):
        self.stop()
        return True

if __name__== "__main__":
    MainApp().run()