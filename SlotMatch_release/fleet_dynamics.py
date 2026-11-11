import numpy as np
import networkx as nx
import sys
import os


from slotmatch_sim.entities import UAV, Carrier, Order, Vehicle
from spatial_metrics import distance_calculator

from enum import Enum

class VehicleState(Enum):
    IDLE = "idle"
    ASSIGNED = "assigned"
    DELIVERING = "delivering"
    CHARGING = "charging"

class RoadNetworkUAV(UAV):
    
    def __init__(self, uav_id, start_node, road_graph, capacity=5, battery_capacity=50, 
                 charging_station_node=None, charging_stations_list=None, speed=120):
        super().__init__(uav_id, start_node, capacity, battery_capacity, 
                         charging_station_node or start_node, speed)
        
        self.current_node = start_node
        self.road_graph = road_graph
        self.charging_station_node = charging_station_node or start_node
        if charging_stations_list is not None:
            self.charging_stations_list = list(charging_stations_list)
        else:
            self.charging_stations_list = [charging_station_node or start_node]
        
        self.vehicle_type = 'uav'
        self.start_grid = start_node
        self.current_grid = start_node
        
        self.battery_capacity = battery_capacity
        self.battery_level = battery_capacity
        self.base_power = 0.32
        self.load_power_factor = 0.08
        self.safety_margin = 0.4
        self.standard_payload = 0.5
        
        self.current_load = 0
        self.orders = []
        self.target_node = None
        self.path = []
        self.path_index = 0

        self.vehicle_state = VehicleState.IDLE
        self.assigned_order = None
        self.pickup_deadline = None
        self.delivery_deadline = None
        self.energy_debug = True
        
        self.total_energy_consumed = 0.0
        self.total_flight_time = 0.0
        self.completed_orders_count = 0
        
        self.in_flight = False
        self.flight_target_node = None
        self.flight_start_node = None
        self.remaining_flight_timesteps = 0
        self.flight_payload = 0.0
        self.flight_total_energy = 0.0

        self.is_flying_to_charge = False

    def calculate_flight_time_to_node(self, target_node):
        dist_km = distance_calculator.get_straight_line_distance(self.current_node, target_node)
        return dist_km / self.speed

    def calculate_enhanced_energy_consumption(self, flight_time_hours, payload_kg=0.0):
        base_energy = flight_time_hours * self.base_power
        load_energy = flight_time_hours * self.load_power_factor * payload_kg
        total_energy = base_energy + load_energy
        return total_energy

    def can_complete_three_phase_mission(self, order):
        if not hasattr(order, 'start_node') or not hasattr(order, 'end_node'):
            return False, "Missing node information"
        
        pickup_time = self.calculate_flight_time_to_node(order.start_node)
        pickup_energy = self.calculate_enhanced_energy_consumption(pickup_time, 0.0)
        
        delivery_time = distance_calculator.get_straight_line_distance(order.start_node, order.end_node) / self.speed
        payload = getattr(order, 'weight', self.standard_payload)
        delivery_energy = self.calculate_enhanced_energy_consumption(delivery_time, payload)
        
        last_nearest_station = self.find_nearest_charging_station(order.end_node)
        return_time = distance_calculator.get_straight_line_distance(order.end_node, last_nearest_station) / self.speed
        return_energy = self.calculate_enhanced_energy_consumption(return_time, 0.0)
        
        total_energy_required = pickup_energy + delivery_energy + return_energy
        total_time = pickup_time + delivery_time + return_time
        
        safe_energy_required = total_energy_required
        
        feasible = self.battery_level >= safe_energy_required
        reason = "Feasible" if feasible else f"Insufficient energy ({self.battery_level:.1f} < {safe_energy_required:.1f} kWh)"
        
        return feasible, reason

    def execute_flight_phase(self, target_node, payload_kg=0.0, phase_name="Flight"):
        if target_node == self.current_node:
            return True, "Already at target"
        
        if self.in_flight:
            return self._update_flight_progress()
        
        flight_time = self.calculate_flight_time_to_node(target_node)
        energy_cost = self.calculate_enhanced_energy_consumption(flight_time, payload_kg)
        
        if self.battery_level < energy_cost:
            return False, f"Insufficient energy ({self.battery_level:.1f} < {energy_cost:.1f} kWh)"
        
        timesteps_needed = max(1, round(flight_time * 6))
        
        self.in_flight = True
        self.flight_start_node = self.current_node
        self.flight_target_node = target_node
        self.remaining_flight_timesteps = timesteps_needed
        self.flight_payload = payload_kg
        self.flight_total_energy = energy_cost
        
        self.battery_level -= energy_cost
        self.total_energy_consumed += energy_cost
        self.total_flight_time += flight_time
        
        if timesteps_needed > 1:
            self.remaining_flight_timesteps -= 1
            return "flying", f"Flying, remaining {self.remaining_flight_timesteps} timesteps"
        else:
            return self._complete_flight()

    def _update_flight_progress(self):
        self.remaining_flight_timesteps -= 1
        
        if self.remaining_flight_timesteps <= 0:
            return self._complete_flight()
        else:
            return "flying", f"Flying, remaining {self.remaining_flight_timesteps} timesteps"

    def _complete_flight(self):
        old_node = self.current_node
        self.current_node = self.flight_target_node
        self.current_grid = self.flight_target_node
        
        self.in_flight = False
        self.flight_target_node = None
        self.flight_start_node = None
        self.flight_payload = 0.0
        self.flight_total_energy = 0.0
        
        return True, "Flight completed"

    def find_nearest_charging_station(self, find_node=None):
        if not self.charging_stations_list:
            return self.charging_station_node
        
        if find_node is None:
            current_pos = self.current_node
        else:
            current_pos = find_node

        min_distance = float('inf')
        nearest_station = self.charging_stations_list[0]
        
        for station in self.charging_stations_list:
            distance = distance_calculator.get_straight_line_distance(current_pos, station)
            if distance < min_distance:
                min_distance = distance
                nearest_station = station
        
        return nearest_station

    def update_charging_station(self):
        nearest = self.find_nearest_charging_station()
        if nearest != self.charging_station_node:
            self.charging_station_node = nearest
        return self.charging_station_node

    def needs_return_to_charge(self, next_order=None):
        battery_ratio = self.battery_level / self.battery_capacity
        if battery_ratio <= self.safety_margin:
            return True, f"Battery below safety threshold ({battery_ratio*100:.1f}% <= {self.safety_margin*100}%)"
        
        return False, "Battery sufficient"

    def _perform_charging(self):
        old_level = self.battery_level
        self.battery_level = self.battery_capacity
        self.vehicle_state = VehicleState.IDLE

    def return_to_charging_station(self):
        self.update_charging_station()

        if self.current_node == self.charging_station_node:
            self._perform_charging()
            self.is_flying_to_charge = False
            return True
        
        if self.in_flight and not self.is_flying_to_charge:
            self.in_flight = False
            self.flight_target_node = None
        
        if self.is_flying_to_charge:
            if self.in_flight:
                result, message = self.execute_flight_phase(self.charging_station_node, 0.0, "Return to charge")
                if result == True:
                    self._perform_charging()
                    self.is_flying_to_charge = False
                    return True
                elif result == "flying":
                    return "flying_to_charge"
                else:
                    self.is_flying_to_charge = False
                    return False
            else:
                self._perform_charging()
                self.is_flying_to_charge = False
                return True
        
        result, message = self.execute_flight_phase(self.charging_station_node, 0.0, "Return to charge")
        
        if result == True:
            self._perform_charging()
            return True
        elif result == "flying":
            self.is_flying_to_charge = True
            return "flying_to_charge"
        else:
            return False

    def can_deliver_order_safely(self, order):
        can_complete, reason = self.can_complete_three_phase_mission(order)
        return can_complete

    def get_energy_statistics(self):
        if self.completed_orders_count > 0:
            avg_energy_per_order = self.total_energy_consumed / self.completed_orders_count
            avg_time_per_order = self.total_flight_time / self.completed_orders_count
        else:
            avg_energy_per_order = 0
            avg_time_per_order = 0
            
        return {
            'battery_level': self.battery_level,
            'battery_percentage': self.battery_level / self.battery_capacity * 100,
            'total_energy_consumed': self.total_energy_consumed,
            'total_flight_time': self.total_flight_time,
            'completed_orders': self.completed_orders_count,
            'avg_energy_per_order': avg_energy_per_order,
            'avg_time_per_order': avg_time_per_order,
            'needs_charging': self.needs_return_to_charge()[0]
        }

class RoadNetworkCarrier:
    
    def __init__(self, carrier_id, start_node, road_graph, capacity=float('inf'), 
                 speed=90, range_limit=50, max_work_hours=8):
        self.vehicle_id = carrier_id
        self.vehicle_type = 'carrier'
        self.start_node = start_node
        self.current_node = start_node
        self.road_graph = road_graph
        
        self.capacity = capacity
        self.remaining_road_timesteps = 0
        self.is_moving_on_road = False
        self.road_movement_target = None
        
        self.speed = speed
        self.range_limit = range_limit
        
        self.max_work_hours = max_work_hours
        self.worked_hours = 0.0
        
        self.current_load = 0
        self.start_grid = start_node
        self.current_grid = start_node
    
    def calculate_road_travel_time(self, from_node, to_node):
        if from_node == to_node:
            return 0.0
        try:
            distance_km = distance_calculator.get_road_network_distance(from_node, to_node)
            travel_time = distance_km / self.speed
            return travel_time
        except Exception as e:
            return float("inf")
        
    def execute_road_movement_phase(self, target_node, phase_name="Road movement"):
        if not self.is_moving_on_road:
            travel_time = self.calculate_road_travel_time(self.current_node, target_node)
            
            if travel_time == float("inf"):
                return False, f"Cannot reach target node {target_node}"
            
            timesteps_needed = max(1, round(travel_time * 6))
            
            self.is_moving_on_road = True
            self.road_movement_target = target_node
            self.remaining_road_timesteps = timesteps_needed
        
        self.remaining_road_timesteps -= 1
        
        if self.remaining_road_timesteps <= 0:
            self.current_node = self.road_movement_target
            self.is_moving_on_road = False
            self.road_movement_target = None
            return True, f"{phase_name} completed"
        else:
            return "moving", f"{phase_name} in progress, remaining {self.remaining_road_timesteps} timesteps"
    
    def update_work_time(self, additional_hours):
        self.worked_hours += additional_hours
    
    def is_available_for_work(self):
        return self.worked_hours < self.max_work_hours
    
    def get_remaining_work_time(self):
        return max(0, self.max_work_hours - self.worked_hours)
    
    def reset_work_time(self):
        self.worked_hours = 0.0


class RoadNetworkOrder:
    
    def __init__(self, order_id, start_node, end_node, start_time, deadline, 
                 weight=0.5, created_time=0):
        self.order_id = order_id
        self.start_node = start_node
        self.end_node = end_node
        self.start_time = start_time
        self.deadline = deadline
        self.weight = weight
        self.created_time = created_time
        
        self.start_grid = start_node
        self.end_grid = end_node
