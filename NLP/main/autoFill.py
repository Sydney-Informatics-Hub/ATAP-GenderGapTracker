#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is a modified code from the below GitHub page

Source: https://github.com/germanesosa/ipywidget-autocomplete
"""
from ipywidgets import *

def autoFill(opts=[''], val='',txt='',placehold='Please type here...',callback=False):
    opts.append('')
    
    def dropFunc(value):
        if (value.new in opts):
            dropClose()
            if (callable(callback)):
                callback(value)
        text.value = value.new                
    
    def dropClose():
        drop.layout.visibility='hidden'
        selDropBox.layout.visibility='hidden'            
        selDropBox.layout.display='none'         
    
    def textFunc(value):
        matched = False
        if (len(value.new)>len(value.old)):
            if (len(value.new)>=0):
                word = value.new
                out = [word]
                for mystring in opts:
                    if word.lower() in mystring.lower(): 
                        if (mystring.lower()==word.lower()):
                            matched = True
                        out.append(mystring)
                if (not matched):
                    drop.layout.visibility='visible'
                    selDropBox.layout.visibility='visible'
                    selDropBox.layout.display='flex'
                    out.append('')
                    drop.options=out
                
    drop = Select(
                options=opts,
                value=val,
                description=txt,
                disabled=False,
                layout = widgets.Layout(width='150px')
           )     
    
    text = Text(
                value=val,
                placeholder=placehold,
                description=txt,
                disabled=False,
                layout = widgets.Layout(width='150px', height='35px')
            )         
    
    drop.observe(dropFunc, names='value')
    text.observe(textFunc, names='value')
    selTextBox = Box([text])
    selDropBox = Box([drop], layout = Layout(display='none', 
                                             top='3px', 
                                             visibility='hidden', 
                                             flex_flow='column', 
                                             width='150px'))
    
    vbox = VBox([selTextBox, selDropBox],layout = Layout(display='flex', 
                                                         flex_flow='column',
                                                         width='170px'))
    
    return vbox