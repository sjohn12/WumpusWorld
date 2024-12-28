# Python version: 3.11
#Built off sample assignment 2 solution posted

import pomegranate as pg
import math
import copy
from engine import Environment, Action, Percept, Orientation, Location, GameConfig, Agent
from typing import List, Tuple, Dict, Any, Optional, Union, Union, TypedDict, Set, Deque
from dataclasses import dataclass
import random
from collections import deque
import heapq
from utils import manhattan_distance


class ProbAgent(Agent):
    def __init__(
            self,
            game_config: GameConfig,
    ):
        """Initialize the agent.

        Args:
            game_config (GameConfig): The game configuration.
        """
        super().__init__(game_config)
        # keep track of the visited locations (tuple of (x, y))
        self.visited_locations: Set[Tuple[int, int]] = set()
        
        #keep track of stench locations
        self.stench_locations: Set[Tuple[int, int]] = set()
        #keep track of breeze locations
        self.breeze_locations: Set[Tuple[int, int]] = set()
        
        #keep track if agent heard scream
        self.heard_scream: bool = False
        
        self.pit_prior = 0.2
        self.wumpus_prior = 0.1
        self.pit_breeze_graph = pg.FactorGraph()
        self.wumpus_stench_graph = pg.FactorGraph()
        
        
        # the escape path, containing locations ordered from the next location to the escape location
        self.escape_path: List[Tuple[int, int]] = []
        # the tie-breaking turn action (to break the tie when the agent has multiple actions with the same priority); the agent has a 50% chance to be left-handed or right-handed at birth
        self.tie_breaking_turn: Action = Action.LEFT if random.random() < 0.5 else Action.RIGHT
        # remember the last action (to avoid repeated turning in the same direction)
        self.last_action: Optional[Action] = None
        self.escaping: bool = False
        print("Agent is ready to explore the cave and find the gold...")
        print("Brain power:", "NO BRAIN")
        print()

    def choose_random_turn_action(self) -> Action:
        """Choose a random turn action (left or right) to break the tie when the agent has multiple actions with the same priority.

        Returns:
            Action: The random turn action (left or right).
        """
        # The agent has a 50% chance to be left-handed or right-handed at birth
        return Action.LEFT if random.random() < 0.5 else Action.RIGHT

    def choose_action(self, percept: Percept, env: Environment) -> Action:
        """Choose the next action to take based on the current percept.

        Action logic:
        Two special cases:

            1. If the agent has gold and is at the escape location, climb.

            2. If the agent perceives glitter and does not have gold, grab.

        In the rest cases, I'm breaking down this problem into two different sub-problems to solve:

            1. Explore the cave and find the gold (by moving randomly)

            2. Escape from the cave after getting the gold:
                a. generate an escape plan using A* search (based on the visited locations and the escape location, with the goal to find the shortest path, with manhattan distance and foreseeable action cost as the heuristic function)
                b. execute the escape plan (move towards the target location by taking the right action step by step)

        Args:
            percept (Percept): The current percept.
            env (Environment): The environment.

        Returns:
            Action: The next action to take.
        """
        current_location: Location = env.agent_location
        current_orientation: Orientation = env.agent_orientation
        # Update the visited locations
        self.update_visited_locations(current_location)
        
        if percept.stench:
            self.update_stench_locations(current_location)
            for i in self.stench_locations:
                if manhattan_distance((env.wumpus_location.x,env.wumpus_location.y), (i.x,i.y)) == 1:
                        self.wumpus_stench_graph.add_edge((f"Wumpus_{env.wumpus_location.x}_{env.wumpus_location.y}", f"Stench_{i.x}_{i.y}"))
            wumpus_prob = self.wumpus_stench_graph.nodes[0].marginal()[1]
        if percept.breeze:
            self.update_breeze_locations(current_location)
            for i in env.pit_locations:
                for j in self.breeze_locations:
                    if manhattan_distance((i.x,i.y), (j.x,j.y)) == 1:
                        self.pit_breeze_graph.add_edge((f"Pit_{i.x}_{i.y}", f"Breeze_{j.x}_{j.y}"))
            pit_prob = self.wumpus_stench_graph.nodes[0].marginal()[1]
            
        if percept.scream:
            self.heard_scream = True

        # If the agent is escaping from the cave, execute the escape plan
        if self.escaping:
            return self.escape_from_cave(env)

        # If the agent has gold and is at the escape location, climb
        if percept.agent_has_gold and current_location.is_location(env.agent_escape_location):
            return Action.CLIMB

        # If the agent perceives glitter and does not have gold, grab
        elif percept.glitter and not percept.agent_has_gold:
            self.escape_path = []
            return Action.GRAB

        # If the agent has gold, generate an escape plan and execute it
        elif percept.agent_has_gold:
            self.escaping = True
            print("Yo! I've got the gold! Fleeeeeeeeee!")
            return self.escape_from_cave(env)
        
        elif pit_prob > 0.5 or wumpus_prob > 0.5 or current_location in self.breeze_locations or current_location in self.stench_locations:
            current_location = self.update_visited_locations[-2]
            return Action.less_random()

        # If the agent does not have gold, randomly explore the cave and find the gold
        else:
            return Action.less_random()

    def escape_from_cave(self, env: Environment) -> Action:
        """Escape from the cave after getting the gold, or without the gold if the agent is forced to escape.

        Args:
            env (Environment): The environment.

        Returns:
            Action: The next action to take.
        """
        current_location: Location = env.agent_location
        current_orientation: Orientation = env.agent_orientation
        if current_location == env.agent_escape_location:
            return Action.CLIMB

        if not self.escape_path:
            # If the escape path is not generated yet, generate it
            self.escape_path: List[Tuple[int, int]] = self.find_shortest_path_to_destination_among_valid_locations(
                start=current_location.tuple,
                destination=env.agent_escape_location.tuple,
                valid_locations=self.visited_locations,
                start_orientation=current_orientation,
                hypothetical_world_size=env.world_size,  # in this dumb version, the agent knows the true world size without any uncertainty
            )

        if self.escape_path:
            # If the escape path is already ready generated, execute it. As the agent is moving towards the escape location, the escape path will be updated to only contain the remaining locations to move to.
            next_location = self.escape_path[0]
            next_action = self.move_towards_target(
                target_location=next_location,
                current_location=current_location,
                env=env,
            )
            # Update the escape path
            if next_action == Action.FORWARD:
                self.escape_path.pop(0)
            return next_action
        else:
            raise ValueError('The escape path is empty.')

    @staticmethod
    def find_shortest_path_to_destination_among_valid_locations(
            start: Tuple[int, int],
            destination: Tuple[int, int],
            valid_locations: Set[Tuple[int, int]],
            start_orientation: Optional[Orientation] = None,
            hypothetical_world_size: Optional[Tuple[int, int]] = (math.inf, math.inf),
            include_start: bool = False,
    ) -> List[Tuple[int, int]]:
        """Find the shortest path from start to destination within the valid locations using BFS and A* search.

        Args:
            start (Tuple[int, int]): The start location.
            destination (Tuple[int, int]): The destination location.
            valid_locations (Set[Tuple[int, int]]): The valid locations.
            start_orientation (Optional[Orientation]): The start orientation of the agent. If provided, the orientation will be considered in the cost function when using A* search (because the agent can only move forward or turn left or right at one step, and each turn costs 1 action).
            hypothetical_world_size (Optional[Tuple[int, int]]): The world size perceived by the agent. If provided, the world size will be used to restrict the neighbour locations during path finding.
            include_start (bool): Whether to include the start location in the path.

        Returns:
            List[Tuple[int, int]]: The shortest path from start to destination, consisting of the locations going from start to destination.

        Example:
            Current map (agent is at (2, 2), and the destination is (0, 0)):
            |.   |.   |.   |.   |
            |.   |    |v  G|    |
            |.   |    |    |    |
            |.   |    |    |    |

            Escape route (not including the agent's start location):
             [(2, 3), (1, 3), (0, 3), (0, 2), (0, 1), (0, 0)]
        """
        # the priority queue for A* search
        queue: List[Tuple[int, Tuple[int, int]]] = []
        # push the start location into the queue with priority 0 (the priority is the cost so far plus the manhattan distance from the start to the destination); 0 means the start location has the highest priority
        heapq.heappush(queue, (0, start))
        # store the parent of each location during the search from the start to the destination (the shortest path)
        came_from = {}
        # store the cost so far during the search from the start to the current location (the number of steps from the start to the current location); each key-value pair is {location: cost_so_far}
        cost_so_far: Dict[Tuple[int, int], int] = {}
        # the start location has no parent
        came_from[start]: Optional[Tuple[int, int]] = None
        # the cost so far from the start to the start is 0
        cost_so_far[start]: int = 0
        # current orientation
        current_orientation = copy.deepcopy(start_orientation)

        while queue:
            # While the queue is not empty, get the location with the lowest priority from the queue and explore its neighbours to find the shortest path from the start to the destination (A* search)
            current_location = heapq.heappop(queue)[1]
            if current_location == destination:
                # If the current location is the destination, break the loop
                break

            # For each neighbour of the current location, if the neighbour is a valid location, update the cost so far and the priority, and push the neighbour into the queue
            for neighbour in Location(x=current_location[0], y=current_location[1], world_size=hypothetical_world_size).neighbours():
                neighbour = neighbour.tuple
                # If the neighbour is a valid location, update the cost so far and the priority, and push the neighbour into the queue
                if neighbour in valid_locations:
                    # The cost means the number of steps from the start to the current location (each movement costs 1 action)
                    new_cost = cost_so_far[current_location] + 1
                    # If the orientation is taken into account, there will be an additional cost for turning left or right at each step
                    if start_orientation is not None:
                        # turning left or right costs 1 action, and turning 180 degrees costs 2 actions
                        new_cost += Orientation.get_number_of_turns_required_to_face_location(
                            start=current_location,
                            destination=neighbour,
                            start_orientation=current_orientation,
                        )
                        # Update the current orientation
                        current_orientation = Orientation.get_orientation_facing_location(
                            start=current_location,
                            destination=neighbour,
                        )

                    if neighbour not in cost_so_far or new_cost < cost_so_far[neighbour]:
                        # If the neighbour is not visited yet or the new cost is less than the cost so far, update the cost so far and the priority, and push the neighbour into the queue for further search
                        cost_so_far[neighbour] = new_cost
                        # The priority is the cost so far plus the manhattan distance from the neighbour to the destination (heuristic function)
                        priority = new_cost + manhattan_distance(neighbour, destination)
                        # Push the neighbour into the queue
                        heapq.heappush(queue, (priority, neighbour))
                        # Update the came_from dictionary
                        came_from[neighbour] = current_location

        # Reconstruct the shortest path from the start to the destination
        path: List[Tuple[int, int]] = []
        # Start from the destination and go back to the start to reconstruct the shortest path
        current_location = destination
        while current_location != start:
            # While the current location is not the start location, add the current location to the path and update the current location to its parent
            path.append(current_location)
            # came_from[current] is the parent of the current location
            current_location = came_from[current_location]
        # Add the start location to the path if include_start is True
        if include_start:
            path.append(start)
        # Reverse the path to get the correct order from the start to the destination
        path.reverse()
        return path

    def update_visited_locations(self, location: Location):
        """Update the set of visited locations.

        Args:
            location (Location): The location to add to the set of visited locations.
        """
        self.visited_locations.add(location.tuple)

    def update_stench_locations(self, location: Location):
        """Update the set of stench visited locations.

        Args:
            location (Location): The location to add to the set of visited locations.
        """
        self.stench_locations.add(location.tuple)
        
    def update_breeze_locations(self, location: Location):
        """Update the set of stench visited locations.

        Args:
            location (Location): The location to add to the set of visited locations.
        """
        self.breeze_locations.add(location.tuple)
        
    def move_towards_target(
            self,
            target_location: Tuple[int, int],
            current_location: Location,
            env: Environment
    ) -> Action:
        """Move towards the target location.

        Args:
            target_location (Tuple[int, int]): The target location.
            current_location (Location): The current location.
            env (Environment): The environment.

        Returns:
            Action: The action to take to move towards the target location.
        """
        target_orientation: Orientation = Orientation.get_orientation_facing_location(
            start=current_location.tuple,
            destination=target_location,
        )
        current_orientation = env.agent_orientation
        if current_location.is_location(target_location):
            return Action.CLIMB
        elif target_orientation == current_orientation:
            return Action.FORWARD
        elif target_orientation == current_orientation.turn_left():
            return Action.LEFT
        elif target_orientation == current_orientation.turn_right():
            return Action.RIGHT
        else:
            # If the agent is facing the opposite direction of the target orientation, turn 180 degrees by using the tie-breaking turn action
            return self.tie_breaking_turn

    def run(self):
        env = Environment(
            world_size=self.game_config.world_size,
            agent_start_location=self.game_config.agent_start_location,
            agent_start_orientation=self.game_config.agent_start_orientation,
            agent_escape_location=self.game_config.agent_escape_location,
            agent_has_arrow=self.game_config.agent_has_arrow,
            agent_has_gold=self.game_config.agent_has_gold,
            pit_prob=self.game_config.pit_prob,
            allow_climb_without_gold=self.game_config.allow_climb_without_gold,
            has_wumpus=self.game_config.has_wumpus,
        )
        
        for i in env.pit_locations:
            node = pg.Node(f"Pit_{i.x}_{i.y}", pg.Categorical([1-self.pit_prior, self.pit_prior]))
            self.pit_breeze_graph.add_node(node)
            
        node = pg.Node(f"Wumpus_{env.wumpus_location.x}_{env.wumpus_location.y}", pg.Categorical([1-self.wumpus_prior, self.wumpus_prior]))
        self.wumpus_stench_graph.add_node(node)
            
                    
        cumulative_reward = 0
        percept: Percept = env.get_initial_percept()

        visualize = lambda: env.visualize(
            visualize_paths=(
                (self.visited_locations, '.'),
                (self.escape_path, '+'),
            ),
        )
        while not percept.done:
            visualize()
            print('Percept:', percept)
            action: Action = self.choose_action(percept, env)
            self.last_action = action
            print()
            print('Action:', action)
            percept = env.step(action)
            self.cumulative_reward += percept.reward
        visualize()
        print('Percept:', percept)
        print('Cumulative reward:', self.cumulative_reward)


if __name__ == "__main__":
    game_config = GameConfig(
        world_size=(4, 4),  # (4, 4)
        agent_start_location=(0, 0),
        agent_start_orientation=Orientation.E,
        agent_escape_location=(0, 0),
        agent_has_arrow=True,
        agent_has_gold=False,
        pit_prob=0,   # 0.2
        allow_climb_without_gold=True,
        has_wumpus=True,
    )
    agent = ProbAgent(
        game_config=game_config,
    )
    agent.run()
    
#average score over 1000 runs: 43.21