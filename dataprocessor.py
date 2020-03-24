import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import folium
import pytz as tz  # better alternatives -> Apache arrow or pendulum
from datetime import datetime
from PIL import Image
import urllib
import urllib.request
import wget

class DataProcessor:
    lat_min, lat_max, lon_min, lon_max = 59.1510, 59.6238, 17.5449, 18.6245
    key = "AIzaSyB0LOUHwF7jrKQT3R3rcqzSdeB4ARResuc"
    static_map = "https://maps.googleapis.com/maps/api/staticmap?center=Stockholm&zoom={2}&size={3}x{3}&markers=color:red%7Clabel:C%7C{0},{1}&maptype=roadmap&key=" + key
    def __init__(self, datadir, filename):
        self.datadir = datadir
        self.filename = filename
    

    def _parse_datetime(self, s):
        tzone = tz.timezone("Europe/Stockholm")
        utc = datetime.strptime(s, '%Y-%m-%dT%H:%M:%SZ')
        return tz.utc.localize(utc).astimezone(tzone)

    def load_date(self):
        print("loading data...")
        filename = os.path.join(
            self.datadir, self.filename)
        df = pd.read_csv(filename, sep='\t', header=None)
        df.columns = ['uid', 'timestamp', 'lat', 'lon', 'venue_id']
        return df
    
    def data_process(self, df):
        print("processing data...")
        df['ts'] = df['timestamp'].apply(lambda x: self._parse_datetime(x))
        df = df.drop('timestamp', axis=1, errors='ignore')
        df['date'] = df['ts'].astype(object).apply(lambda x: x.date())
        df['time'] = df['ts'].astype(object).apply(lambda x: x.time())
        return df

    def data_filter(self, df):
        print("filtering data...")
        df = df[(df['lon'] > DataProcessor.lon_min) & (df['lon'] < DataProcessor.lon_max) &
                (df['lat'] > DataProcessor.lat_min) & (df['lat'] < DataProcessor.lat_max)]
        return df


    def _get_map(self, x, y, z, size, filename):
        #static_map = "http://staticmap.openstreetmap.de/staticmap.php?center={0},{1}&zoom={2}&size={3}x{3}&maptype=mapnik".format(y,x,z,size)

        # static_map = "https://maps.googleapis.com/maps/api/staticmap?center=Stockholm&zoom={2}&size={3}x{3}&markers=color:red%7Clabel:C%7C{0},{1}&maptype=roadmap&key=AIzaSyB0LOUHwF7jrKQT3R3rcqzSdeB4ARResuc".format(
        #     x, y, z, size)
        static_map = DataProcessor.static_map.format(x, y, z, size)
        static_map_filename, headers = urllib.request.urlretrieve(
            static_map, filename)
        return static_map_filename


    def geomap(self, data, df, zoom=15, point_size=3, point_color='r', point_alpha=1):
        #corrections to match geo with static map
        z = zoom
        picsize = 1000
        wx = 1.0*360*(picsize/256)/(2**z)
        wy = 0.76*360*(picsize/256)/(2**z)

        #center of manhattan
        y = 18.0649  # lon
        x = 59.33258  # lat

        x_min, x_max = x-wx/2, x+wx/2
        y_min, y_max = y-wy/2, y+wy/2

        static_map_filename = os.path.join(
            self.datadir, 'Stockholm_staticmap_{}_{}.png'.format(x, y, z, picsize))

        if os.path.isfile(static_map_filename) == False:
            self._get_map(x, y, z, picsize, static_map_filename)

        img = Image.open(static_map_filename)

        #add the static map
        plt.imshow(img, zorder=0, extent=[
                   x_min, x_max, y_min, y_max], interpolation='none', aspect='auto')

        #add the scatter plot of events
        plt.plot(
            data['lat'],
            data['lon'],
            '.',
            markerfacecolor=point_color,
            markeredgecolor='k',
            markersize=point_size,
            alpha=point_alpha)

        #limit the plot to the given box
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)




if __name__ == '__main__':
    dataprocessor = DataProcessor(
        '/Users/wangyifan/Google Drive/checkin', 'loc-gowalla_totalCheckins.txt')
    #indexer.build_index('../../reuters/training')
    # start = time.time()
    df = dataprocessor.load_date()
    df = dataprocessor.data_filter(df)
    df = dataprocessor.data_process(df)

    plt.style.use('seaborn-white')
    fig = plt.figure()
    fig.set_size_inches(20, 20)
    dataprocessor.geomap(df[['lat', 'lon']], df=df)
    plt.show()
