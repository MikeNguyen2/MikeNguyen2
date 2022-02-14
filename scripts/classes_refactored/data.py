import os, sys, sqlite3
import time 

class data_manager():
    def __init__(self):
        self.connection = sqlite3.connect("/home/mike/cobot/scripts/classes_refactored/data.db")
        self.cursor = self.connection.cursor()

    def create_table(self):
        try:
            sql = "CREATE TABLE mtp(" \
                "name TEXT, "\
                "length REAL, " \
                "width REAL, " \
                "height REAL, "\
                "well_diameter REAL, " \
                "well_distance REAL, " \
                "well_radius REAL, "\
                "pattern_left REAL, "\
                "pattern_top REAL, "\
                "dist INTEGER, "\
                "canny_limit INTEGER, "\
                "lower_limit INTEGER )"
            self.cursor.execute(sql)
            print("table mtp created")
        except:
            print("table mtp already exists")

        try:
            sql = "CREATE TABLE box(" \
                "name TEXT, "\
                "length REAL, " \
                "width REAL, " \
                "height REAL, "\
                "well_diameter REAL, " \
                "well_distance REAL, " \
                "well_radius REAL, "\
                "pattern_left REAL, "\
                "pattern_top REAL, "\
                "dist INTEGER, "\
                "canny_limit INTEGER, "\
                "lower_limit INTEGER )"
            self.cursor.execute(sql)
            print("table box created")
        except:
            print("table box already exists")

        try:
            sql = "CREATE TABLE phenol(" \
                "name TEXT, "\
                "length REAL, " \
                "width REAL, " \
                "height REAL, "\
                "well_diameter REAL, " \
                "well_distance REAL, " \
                "well_radius REAL, "\
                "pattern_left REAL, "\
                "pattern_top REAL, "\
                "dist INTEGER, "\
                "canny_limit INTEGER, "\
                "lower_limit INTEGER )"
            self.cursor.execute(sql)
            print("table mtp created")
        except:
            print("table phenol already exists")

        try:
            sql = "CREATE TABLE pipette(" \
                "name TEXT, "\
                "length REAL, " \
                "width REAL, " \
                "height REAL, "\
                "rotation.x REAL, " \
                "rotataion.y REAL, " \
                "rotation.z REAL )"
            self.cursor.execute(sql)
            print("table mtp created")
        except:
            print("table phenol already exists")


    def insert(self, table):
        try:
            sql = "INSERT INTO mtp VALUES("\
                "'purple', "\
                "0.085, "\
                "0.124, "\
                "0.015, "\
                "0.007, "\
                "0.0035, "\
                "0.003, "\
                "0.0085, "\
                "0.005, "\
                "10, "\
                "100, "\
                "10)"
            self.cursor.execute(sql)
            self.connection.commit()

            sql = "INSERT INTO mtp VALUES("\
                "'white', "\
                "0.085, "\
                "0.124, "\
                "0.015, "\
                "0.007, "\
                "0.0035, "\
                "0.003, "\
                "0.0085, "\
                "0.005, "\
                "10, "\
                "100, "\
                "10)"
            self.cursor.execute(sql)
            self.connection.commit()

            sql = "INSERT INTO box VALUES("\
                "'orange', "\
                "0.08, "\
                "0.112, "\
                "0.035, "\
                "0.005, "\
                "0.0025, "\
                "0.003, "\
                "0.0085, "\
                "0.005, "\
                "10, "\
                "50, "\
                "10)"
            self.cursor.execute(sql)
            self.connection.commit()
            print("data inserted")
        except:
            print("invalid insertion")

    def get_info(self, table, name="*", attribute="*"):
        try:
            infos = []
            sql = "SELECT "+ attribute +" FROM "+ table +""
            if name != "*":
                sql += " WHERE name = {}".format("\'" + name + "\'")
            self.cursor.execute(sql)
            for info in self.cursor:
                print(info)
                infos.append(info)
            return info
        except:
           print("no information available")

    def update(self, table, parameter, old_value, new_value ):
        try:
            sql = "UPDATE {}".format(table)+" "\
            "SET " + parameter + " = {}".format("\'" + new_value + "\'")
            "WHERE " + parameter + " = {}".format("\'" + old_value + "\'")
            self.cursor.execute(sql)
            self.connection.commit()
            print("data got updated")
        except:
            print("update is invalid")

    def delete_data(self, table, name):
        try:
            sql = "DELETE FROM " + table + " "\
            "WHERE name = {}".format("\'" + name + "\'")
            self.cursor.execute(sql)
            self.connection.commit()
            print("data was deleted")
        except:
            print("data deletion failed")

    def delete_table(self, table):
        try:
            sql = "DROP TABLE " + table
            self.cursor.execute(sql)
            self.connection.commit()
            print("table " + table + " deleted")
        except:
            print("table deletion of " + table + " failed")

    def close(self):
        try:
            self.connection.close()
            print("connection closed")
        except:
            print("connection is already closed")

    def test(self, table, parameter, old_value, new_value ):
        sql = "UPDATE {}".format(table)+" "\
        "SET " + parameter + " = {}".format("\'" + new_value + "\'")
        "WHERE " + parameter + " = {}".format("\'" + old_value + "\'")
        print(sql)
        self.cursor.execute(sql)
       
if __name__ == "__main__":
    connection = sqlite3.connect("/home/mike/cobot/scripts/classes_refactored/data.db")
    cursor = connection.cursor()
    
    my_manager = data_manager()
    my_manager.delete_table('mtp')
    my_manager.create_table()
    my_manager.insert('mtp')
    info = my_manager.get_info('mtp',name='purple', attribute='width')
    my_manager.update('mtp', 'name', 'purple', 'white')
    info2 = my_manager.get_info('mtp')

    #my_manager.update('rec_with_circ', 'name', 'mtp', 'mp')
    #my_manager.get_info('mtp')
    #my_manager.get_info('mp')
    my_manager.close()