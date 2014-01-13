#!/usr/bin/env python

"""
wxpython GUI around matplotlib and psnr/ssim calculations

Fredrik Pihl 2014-01-09
"""

import os
import pprint
import random
import wx
import re
import numpy as np

from ycbcr import YCbCr

# The recommended way to use wx with mpl is with the WXAgg
# backend.
import matplotlib
matplotlib.use('WXAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_wxagg import \
    FigureCanvasWxAgg as FigCanvas, \
    NavigationToolbar2WxAgg as NavigationToolbar


class CalcFrame(wx.Frame):
    """
    The main frame of the application
    """
    title = 'wxPython with matplotlib'

    def __init__(self):
        wx.Frame.__init__(self, None, -1, self.title)

        self.algo_list = [
            'psnr-y',
            'psnr-u',
            'psnr-v',
            'psnr-bd',
            'psnr-all',
            'ssim']

        self.algo_map = {
                'psnr-y': 0,
                'psnr-u': 1,
                'psnr-v': 2,
                'psnr-bd': 3 }

        self.yuv_format_list = ['YV12', 'IYUV', 'UYVY', 'YVYU', 'YUY2']

        self.create_menu()
        self.create_status_bar()
        self.create_main_panel()

    def create_menu(self):
        self.menubar = wx.MenuBar()

        menu_file = wx.Menu()
        m_expt = menu_file.Append(-1, "&Save plot\tCtrl-S", "Save plot to file")
        self.Bind(wx.EVT_MENU, self.on_save_plot, m_expt)
        menu_file.AppendSeparator()
        m_exit = menu_file.Append(-1, "E&xit\tCtrl-X", "Exit")
        self.Bind(wx.EVT_MENU, self.on_exit, m_exit)

        menu_help = wx.Menu()
        m_about = menu_help.Append(-1, "&About\tF1", "About the demo")
        self.Bind(wx.EVT_MENU, self.on_about, m_about)

        self.menubar.Append(menu_file, "&File")
        self.menubar.Append(menu_help, "&Help")
        self.SetMenuBar(self.menubar)

    def create_main_panel(self):
        """
        Creates the main panel with all the controls on it:
            * mpl canvas
            * mpl navigation toolbar
            * Control panel for interaction
        """
        self.panel = wx.Panel(self)

        # Create the mpl Figure and FigCanvas objects.
        # 5x4 inches, 100 dots-per-inch
        self.dpi = 100
        self.fig = Figure((5.0, 4.0), dpi=self.dpi)
        self.canvas = FigCanvas(self.panel, -1, self.fig)

        # Since we have only one plot, we can use add_axes
        # instead of add_subplot, but then the subplot
        # configuration tool in the navigation toolbar wouldn't
        # work.
        self.axes = self.fig.add_subplot(111)

        # Bind the 'pick' event for clicking on one of the bars
        self.canvas.mpl_connect('pick_event', self.on_pick)

        self.t_load_f1 = wx.TextCtrl(
            self.panel,
            size=(200,-1),
            style=wx.TE_PROCESS_ENTER)

        self.t_load_f2 = wx.TextCtrl(
            self.panel,
            size=(200,-1),
            style=wx.TE_PROCESS_ENTER)

        self.t_size = wx.TextCtrl(
            self.panel,
            size=(125,-1),
            style=wx.TE_PROCESS_ENTER)
        self.t_size.SetValue('WIDTHxHEIGHT')

        self.algo = wx.ComboBox(
                self.panel,
                1,
                "psnr-y",
                wx.DefaultPosition,
                (200, -1),
                self.algo_list,
                wx.CB_DROPDOWN)

        self.yuv_format = wx.ComboBox(
                self.panel,
                1,
                "YV12",
                wx.DefaultPosition,
                (200, -1),
                self.yuv_format_list,
                wx.CB_DROPDOWN)

        # buttons
        self.b_load_f1 = wx.Button(self.panel, -1, "Load...")
        self.b_load_f2 = wx.Button(self.panel, -1, "Load...")
        self.b_calc = wx.Button(self.panel, -1, "CALC")
        self.Bind(wx.EVT_BUTTON, self.on_b_load_f1, self.b_load_f1)
        self.Bind(wx.EVT_BUTTON, self.on_b_load_f2, self.b_load_f2)
        self.Bind(wx.EVT_BUTTON, self.on_b_calc, self.b_calc)

        self.cb_grid = wx.CheckBox(self.panel, -1, "Show Grid",
                                   style=wx.ALIGN_RIGHT)
        self.Bind(wx.EVT_CHECKBOX, self.on_cb_grid, self.cb_grid)

        # Create the navigation toolbar, tied to the canvas
        self.toolbar = NavigationToolbar(self.canvas)

        # Layout with box sizers
        self.vbox = wx.BoxSizer(wx.VERTICAL)
        self.vbox.Add(self.canvas, 1, wx.LEFT | wx.TOP | wx.GROW)
        self.vbox.Add(self.toolbar, 0, wx.EXPAND)
        self.vbox.AddSpacer(10)

        flags = wx.ALIGN_LEFT | wx.ALL | wx.ALIGN_CENTER_VERTICAL

        #sb1 = wx.StaticBox(self.panel, label="Original")
        #sb2 = wx.StaticBox(self.panel, label="Filtered")

        self.hbox1 = wx.BoxSizer(wx.HORIZONTAL)
        self.hbox1.Add(self.t_load_f1, 0, border=3, flag=flags)
        self.hbox1.Add(self.b_load_f1, 0, border=3, flag=flags)

        self.hbox1.Add(self.t_load_f2, 0, border=3, flag=wx.EXPAND|wx.ALL)
        self.hbox1.Add(self.b_load_f2, 0, border=3, flag=wx.EXPAND|wx.ALL)

        self.hbox2 = wx.BoxSizer(wx.HORIZONTAL)
        self.hbox2.Add(self.algo, 0, border=3, flag=flags)
        self.hbox2.Add(self.yuv_format, 0, border=3, flag=flags)
        self.hbox2.Add(self.t_size, 0, border=3, flag=flags)
        self.hbox2.Add(self.cb_grid, 0, border=3, flag=flags)

        self.hbox3 = wx.BoxSizer(wx.HORIZONTAL)
        self.hbox3.Add(self.b_calc, 0, border=3, flag=flags)

        self.vbox.Add(self.hbox1, 0, flag = wx.ALIGN_LEFT | wx.TOP)
        self.vbox.Add(self.hbox2, 0, flag = wx.ALIGN_LEFT | wx.TOP)
        self.vbox.Add(self.hbox3, 0, flag = wx.ALIGN_LEFT | wx.TOP)

        self.panel.SetSizer(self.vbox)
        self.vbox.Fit(self)

    def create_status_bar(self):
        """
        A statusbar is nice to have
        """
        self.statusbar = self.CreateStatusBar()
        self.statusbar.SetStatusText('Hello World')

    def on_b_calc(self, event):
        """
        Calculate
        """
        f1 = self.t_load_f1.GetValue()
        f2 = self.t_load_f2.GetValue()
        a = self.algo.GetValue()
        f = self.yuv_format.GetValue()
        s = self.t_size.GetValue()

        w, h = s.split('x')

        # are all parameters ok?
        if not os.path.isfile(f1) or not os.path.isfile(f2):
            print 'faulty inputfiles'
            raise IOError

        if not a in self.algo_list:
            print 'faulty algorithm'
            raise KeyError

        if not f in self.yuv_format_list:
            print 'faulty format'
            raise KeyError

        if not w.isdigit() or not h.isdigit():
            print 'faulty size'
            raise TypeError

        y = YCbCr(width=int(w), height=int(h), filename=f1,
                  yuv_format_in=f, filename_diff=f2, num=20)

        if 'psnr' in a:
            psnr = [p for p in y.psnr()][:-2]
            if a != 'psnr-all':
                ind = np.arange(len(psnr))  # the x locations for the groups
                self.axes.clear()
                self.axes.plot(ind, [i[self.algo_map[a]] for i in psnr],
                               'o-', picker=5)

                self.axes.set_title(os.path.basename(f1) + ' ' + a)
                self.axes.set_ylabel('dB')
                self.axes.set_xlabel('frame')

                self.canvas.draw()

            elif a == 'psnr-all':
                ind = np.arange(len(psnr))  # the x locations for the groups
                self.axes.clear()
                self.axes.plot(ind, [i[0] for i in psnr], 'ko-', label='Y', picker=5)
                self.axes.plot(ind, [i[1] for i in psnr], 'bo-', label='Cb', picker=5)
                self.axes.plot(ind, [i[2] for i in psnr], 'ro-', label='Cr', picker=5)
                self.axes.plot(ind, [i[3] for i in psnr], 'mo-', label='BD', picker=5)

                self.axes.set_title(os.path.basename(f1) + ' ' + a)
                self.axes.legend()
                self.axes.set_ylabel('dB')
                self.axes.set_xlabel('frame')

                self.canvas.draw()

        elif 'ssim' in a:
            ssim = [s for s in y.ssim()][:-2]
            ind = np.arange(len(ssim))  # the x locations for the groups
            self.axes.clear()
            self.axes.plot(ind, ssim, 'o-', picker=5)

            self.axes.set_title(os.path.basename(f1) + ' ' + a)
            self.axes.set_ylabel('index')
            self.axes.set_xlabel('frame')

            self.canvas.draw()
        else:
            print '~tilt'

    def on_cb_grid(self, event):
        """
        Toggle grid
        """
        self.axes.grid(self.cb_grid.IsChecked())
        self.canvas.draw()

    def on_pick(self, event):
        """
        The event received here is of the type
        matplotlib.backend_bases.PickEvent

        It carries lots of information, of which we're using
        only a small amount here.
        """
        #box_points = event.artist.get_bbox().get_points()
        thisline = event.artist
        xdata, ydata = thisline.get_data()
        ind = event.ind

        msg = 'x:%d y:%.4f' % (xdata[ind], ydata[ind])
        self.statusbar.SetStatusText(msg)

#        msg = "data"
#
#        dlg = wx.MessageDialog(
#            self,
#            msg,
#            "Click!",
#            wx.OK | wx.ICON_INFORMATION)
#
#        dlg.ShowModal()
#        dlg.Destroy()

    def on_text_enter(self, event):
        pass

    def on_save_plot(self, event):
        file_choices = "PNG (*.png)|*.png"

        dlg = wx.FileDialog(
            self,
            message="Save plot as...",
            defaultDir=os.getcwd(),
            defaultFile="plot.png",
            wildcard=file_choices,
            style=wx.SAVE)

        if dlg.ShowModal() == wx.ID_OK:
            path = dlg.GetPath()
            self.canvas.print_figure(path, dpi=self.dpi)
            self.flash_status_message("Saved to %s" % path)

    def on_b_load_f1(self, event):
        """ Open a file"""
        dirname = os.getcwd()
        dlg = wx.FileDialog(self, "Choose a file", dirname, "", "*.yuv", wx.OPEN)
        if dlg.ShowModal() == wx.ID_OK:
            filename = dlg.GetFilename()
            dirname = dlg.GetDirectory()

            # Update text-box with filename
            self.t_load_f1.SetValue(os.path.join(dirname, filename))

            # Try to parse widthxheight from filename
            size = re.findall(r'\d+x\d+', filename)
            if size:
                self.t_size.SetValue(size[0])
        dlg.Destroy()

    def on_b_load_f2(self, event):
        """ Open a file"""
        dirname = os.getcwd()
        dlg = wx.FileDialog(self, "Choose a file", dirname, "", "*.yuv", wx.OPEN)
        if dlg.ShowModal() == wx.ID_OK:
            filename = dlg.GetFilename()
            dirname = dlg.GetDirectory()

            # Update text-box with filename
            self.t_load_f2.SetValue(os.path.join(dirname, filename))
        dlg.Destroy()

    def on_exit(self, event):
        self.Destroy()

    def on_about(self, event):
        msg = """wxPython with matplotlib:

         - Calculate PSNR/SSIM
         - Show or hide the grid
         - Export graph as PNG

         Fredrik Pihl 2014
        """
        dlg = wx.MessageDialog(self, msg, "About", wx.OK)
        dlg.ShowModal()
        dlg.Destroy()

    def flash_status_message(self, msg, flash_len_ms=1500):
        self.statusbar.SetStatusText(msg)
        self.timeroff = wx.Timer(self)
        self.Bind(
            wx.EVT_TIMER,
            self.on_flash_status_off,
            self.timeroff)
        self.timeroff.Start(flash_len_ms, oneShot=True)

    def on_flash_status_off(self, event):
        self.statusbar.SetStatusText('')

if __name__ == '__main__':
    app = wx.PySimpleApp()
    app.frame = CalcFrame()
    app.frame.Show()
    app.MainLoop()
