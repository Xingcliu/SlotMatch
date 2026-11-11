import numpy as np
import pickle
import copy

class Vehicle:
    """Base vehicle class"""
    def __init__(self, vehicle_id, vehicle_type, start_grid, capacity, speed, range_limit):
        self.vehicle_id = vehicle_id
        self.vehicle_type = vehicle_type
        self.start_grid = start_grid
        self.capacity = capacity
        self.speed = speed
        self.range_limit = range_limit
        self.current_grid = start_grid
        self.load = 0
        self.available_time = 0
        self.assigned_orders = []

class Carrier(Vehicle):
    """Ground carrier with time constraints"""
    def __init__(self, carrier_id, start_grid, capacity=float('inf'), speed=30, range_limit=50, 
                 max_work_hours=8):
        super().__init__(carrier_id, 'carrier', start_grid, capacity, speed, range_limit)
        self.max_work_hours = max_work_hours
        self.worked_hours = 0
        
    def can_accept_order(self, estimated_time_hours, order_weight=None):
        time_available = (self.worked_hours + estimated_time_hours) <= self.max_work_hours
        return time_available
        
    def update_work_time(self, additional_hours):
        self.worked_hours += additional_hours
        
    def is_available(self):
        return self.worked_hours < self.max_work_hours

class UAV(Vehicle):
    """UAV with battery constraints"""
    def __init__(self, uav_id, start_grid, capacity=5, battery_capacity=100, 
                 charging_station_grid=None, speed=60):
        super().__init__(uav_id, 'uav', start_grid, capacity, speed, range_limit=float('inf'))
        
        self.battery_capacity = battery_capacity
        self.battery_level = battery_capacity
        self.charging_station_grid = charging_station_grid or start_grid
        
        self.base_power = 2.0
        self.load_power_factor = 0.5
        
    def calculate_energy_consumption(self, flight_time_hours, payload_kg=0):
        base_energy = flight_time_hours * self.base_power
        load_energy = flight_time_hours * self.load_power_factor * payload_kg
        return base_energy + load_energy
        
    def can_complete_mission(self, flight_time_hours, payload_kg):
        mission_energy = self.calculate_energy_consumption(flight_time_hours, payload_kg)
        return_energy = self.calculate_energy_consumption(flight_time_hours * 0.5, 0)
        total_required_energy = mission_energy + return_energy
        return self.battery_level >= total_required_energy * 1.1
        
    def get_max_flight_time_with_payload(self, payload_kg):
        if payload_kg > self.capacity:
            return 0
        power_consumption = self.base_power + self.load_power_factor * payload_kg
        available_energy = self.battery_level * 0.6
        return available_energy / power_consumption
        
    def update_battery_after_flight(self, flight_time_hours, payload_kg):
        consumed_energy = self.calculate_energy_consumption(flight_time_hours, payload_kg)
        self.battery_level = max(0, self.battery_level - consumed_energy)
        
    def needs_charging(self, threshold=0.3):
        return self.battery_level <= self.battery_capacity * threshold
        
    def can_reach_charging_station(self):
        return self.battery_level >= self.battery_capacity * 0.3

class Driver():
    def __init__(self, did, gid, grid):
        self.did = did
        self.gid = gid
        self.grid = grid
        self.grid.driver_list.append(self.did)

    def update(self, Grids):
        self.gid = self.order.end_grid
        self.order = None
        self.grid = Grids[self.gid]
        self.grid.driver_list.append(self.did)

class Order:
    def __init__(self, order_idx, start_time, end_time, start_grid, end_grid, weight=1, priority=1):
        self.order_idx = order_idx
        self.start_time = start_time
        self.end_time = end_time
        self.start_grid = start_grid
        self.end_grid = end_grid
        self.weight = weight
        self.priority = priority
        self.deadline = start_time + 120
        self.status = 'pending'
        self.assigned_vehicle = None
        self.pickup_time = None
        self.delivery_time = None
        self.price = priority * 10
        
    def get_estimated_delivery_time(self, vehicle_speed_kmh=30, grid_distance_km=2):
        return grid_distance_km / vehicle_speed_kmh
        
    def is_urgent(self, current_time):
        time_left = self.deadline - current_time
        return time_left <= 30

class Grid():
    def __init__(self, idx, gps_list):
        self.idx = idx
        self.gps_list = copy.deepcopy(gps_list)

        lons = [x for id, x in enumerate(gps_list) if id % 2 == 0]
        lats = [x for id, x in enumerate(gps_list) if id % 2 == 1]

        self.center = [np.mean(lons), np.mean(lats)]

        self.order_dict = None
        self.driver_list = []
        self.has_uav_landing = False
