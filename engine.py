# Python version: 3.11
#Built off sample assignment 2 solution posted

from typing import List, Tuple, Dict, Union, Optional, Any, Set
from enum import Enum
import random
import copy
from utils import manhattan_distance
from dataclasses import dataclass


class Agent:
    def __init__(self, game_config: 'GameConfig'):
        self.game_config = game_config
        print(f" Wumpus World "
              f"\n{self.game_config}"
              f"\n"
              )
        self.cumulative_reward: int = 0

    def choose_action(self):
        return Action.random()

    def run(self):
        env = Environment(
            world_size=(4, 4),
            agent_start_location=(0, 0),
            agent_start_orientation=Orientation.E,
            agent_has_arrow=True,
            agent_has_gold=False,
            pit_prob=0.2,
            allow_climb_without_gold=False,
            has_wumpus=True,
        )
        percept: Percept = env.get_initial_percept()
        while not percept.done:
            env.visualize()
            print('Percept:', percept)
            action = self.choose_action()
            print()
            print('Action:', action)
            percept = env.step(action)
            self.cumulative_reward += percept.reward
        env.visualize()
        print('Percept:', percept)
        print('Cumulative reward:', self.cumulative_reward)


class Percept():
    time_step: int
    bump: bool
    breeze: bool
    stench: bool
    scream: bool
    glitter: bool
    agent_has_gold: bool
    reward: int
    done: bool

    def __init__(self, time_step: int, bump: bool, breeze: bool, stench: bool, scream: bool, glitter: bool, agent_has_gold: bool, reward: int,
                 done: bool):
        self.time_step = time_step
        self.bump = bump
        self.breeze = breeze
        self.stench = stench
        self.scream = scream
        self.glitter = glitter
        self.agent_has_gold = agent_has_gold
        self.reward = reward
        self.done = done

    def __str__(self):
        return f'time:{self.time_step}: bump:{self.bump}, breeze:{self.breeze}, stench:{self.stench}, scream:{self.scream}, glitter:{self.glitter}, has_gold: {self.agent_has_gold}, reward:{self.reward}, done:{self.done}'


class Action(Enum):
    LEFT = 0
    RIGHT = 1
    FORWARD = 2
    GRAB = 3
    SHOOT = 4
    CLIMB = 5

    @staticmethod
    def random() -> 'Action':
        return random.choice(list(Action))

    @staticmethod
    def less_random() -> 'Action':
        return random.choice([Action.LEFT, Action.RIGHT, Action.FORWARD])

    @staticmethod
    def from_int(n: int) -> 'Action':
        return Action(n)

    def __str__(self):
        symbol = ''
        match self:
            case Action.LEFT:
                symbol = '⬅️'
            case Action.RIGHT:
                symbol = '➡️'
            case Action.FORWARD:
                symbol = '🦵'
            case Action.GRAB:
                symbol = '🤑'
            case Action.SHOOT:
                symbol = '💥'
            case Action.CLIMB:
                symbol = '🪜'
        return f"{self.name} {symbol}"


class Orientation(Enum):
    E = 0
    S = 1
    W = 2
    N = 3

    def symbol(self) -> str:
        match self:
            case Orientation.E:
                return '→'
            case Orientation.S:
                return '↓'
            case Orientation.W:
                return '←'
            case Orientation.N:
                return '↑'

    def turn_right(self) -> 'Orientation':
        match self:
            case Orientation.E:
                return Orientation.S
            case Orientation.S:
                return Orientation.W
            case Orientation.W:
                return Orientation.N
            case Orientation.N:
                return Orientation.E

    def turn_left(self) -> 'Orientation':
        match self:
            case Orientation.E:
                return Orientation.N
            case Orientation.N:
                return Orientation.W
            case Orientation.W:
                return Orientation.S
            case Orientation.S:
                return Orientation.E

    def turn_around(self) -> 'Orientation':
        match self:
            case Orientation.E:
                return Orientation.W
            case Orientation.W:
                return Orientation.E
            case Orientation.N:
                return Orientation.S
            case Orientation.S:
                return Orientation.N

    @staticmethod
    def get_orientation_facing_location(
            start: Union['Location', Tuple[int, int]],
            destination: Union['Location', Tuple[int, int]],
    ) -> 'Orientation':
        """Gets the orientation facing the destination location from the start location.

        Args:
            start (Union[Location, Tuple[int, int]]): The starting location.
            destination (Union[Location, Tuple[int, int]]): The destination location.

        Returns:
            Orientation: The orientation between the two locations.
        """
        location1 = Location(start[0], start[1]) if isinstance(start, tuple) else start
        location2 = Location(destination[0], destination[1]) if isinstance(destination, tuple) else destination
        if location1.is_left_of(location2):
            return Orientation.E
        elif location1.is_right_of(location2):
            return Orientation.W
        elif location1.is_above(location2):
            return Orientation.S
        elif location1.is_below(location2):
            return Orientation.N
        else:
            raise ValueError(f'Start {location1} destination {location2} are not adjacent')

    @staticmethod
    def get_number_of_turns_required_between_orientations(
            start_orientation: 'Orientation',
            destination_orientation: 'Orientation',
    ) -> int:
        """Gets the number of turns required to get from the current orientation to the destination orientation (because the agent can only turn left or right at any step).

        Args:
            destination (Orientation): The destination orientation.

        Returns:
            int: The number of turns required to get from the current orientation to the destination orientation.
        """
        if start_orientation == destination_orientation:
            # No turn required
            return 0
        elif start_orientation == destination_orientation.turn_around():
            # Turn around requires 2 turns
            return 2
        else:
            # Turn left or right
            return 1


    @staticmethod
    def get_number_of_turns_required_to_face_location(
            start: Union['Location', Tuple[int, int]],
            destination: Union['Location', Tuple[int, int]],
            start_orientation: 'Orientation',
    ) -> int:
        """Gets the number of turns required to face a location.

        Args:
            start (Union[Location, Tuple[int, int]]): The starting location.
            destination (Union[Location, Tuple[int, int]]): The destination location.
            start_orientation (Orientation): The starting orientation.

        Returns:
            int: The number of turns required to face the location.
        """
        destination_orientation = Orientation.get_orientation_facing_location(start, destination)
        return Orientation.get_number_of_turns_required_between_orientations(start_orientation, destination_orientation)


class Location:
    x: int
    y: int
    world_size: Tuple[int, int]

    def __init__(
            self,
            x: int = 0,
            y: int = 0,
            world_size: Optional[Tuple[int, int]] = None,
    ):
        """Initializes a location with a given x and y value. The world size is used to determine if a location is at the edge of the world.

        Args:
            x (int): The x value of the location. 0 is at the left. Defaults to 0.
            y (int): The y value of the location. 0 is at the bottom. Defaults to 0.
            world_size (Tuple[int, int], optional): The size of the world in the x and y direction. Defaults to (4, 4).
        """
        self.x = x
        self.y = y
        self.world_size = world_size

    @property
    def tuple(self) -> Tuple[int, int]:
        return self.x, self.y

    def __getitem__(self, index: int) -> int:
        return self.tuple[index]

    def __eq__(self, other:Union['Location', Tuple[int, int]]) -> bool:
        if isinstance(other, Location):
            return self.x == other.x and self.y == other.y
        else:
            return self.x == other[0] and self.y == other[1]

    def __ne__(self, other: Union['Location', Tuple[int, int]]) -> bool:
        return not self.__eq__(other)

    def __contains__(self, other: Union['Location', Tuple[int, int]]) -> bool:
        return self.__eq__(other)

    def __hash__(self) -> int:
        return hash(*self.tuple)

    def __lt__(self, other: Union['Location', Tuple[int, int]]) -> bool:
        if isinstance(other, Location):
            return self.tuple < (other.x, other.y)
        else:
            return self.tuple < other

    def __le__(self, other: Union['Location', Tuple[int, int]]) -> bool:
        if isinstance(other, Location):
            return self.tuple <= (other.x, other.y)
        else:
            return self.tuple <= other

    def __gt__(self, other: Union['Location', Tuple[int, int]]) -> bool:
        if isinstance(other, Location):
            return self.tuple > (other.x, other.y)
        else:
            return self.tuple > other

    def __ge__(self, other: Union['Location', Tuple[int, int]]) -> bool:
        if isinstance(other, Location):
            return self.tuple >= (other.x, other.y)
        else:
            return self.tuple >= other

    @property
    def world_width(self) -> int:
        """The width of the world.

        Returns:
            int: The width of the world.
        """
        return self.world_size[0]

    @property
    def world_height(self) -> int:
        """The height of the world.

        Returns:
            int: The height of the world.
        """
        return self.world_size[1]

    @property
    def max_x(self) -> int:
        """The maximum x value of the world.

        Returns:
            int: The maximum x value of the world.
        """
        return self.world_width - 1

    @property
    def max_y(self) -> int:
        """The maximum y value of the world.

        Returns:
            int: The maximum y value of the world.
        """
        return self.world_height - 1

    def __str__(self):
        return f'({self.x}, {self.y})'

    def is_left_of(self, location: 'Location') -> bool:
        return self.x < location.x and self.y == location.y

    def is_right_of(self, location: 'Location') -> bool:
        return self.x > location.x and self.y == location.y

    def is_above(self, location: 'Location') -> bool:
        return self.y > location.y and self.x == location.x

    def is_below(self, location: 'Location') -> bool:
        return self.y < location.y and self.x == location.x

    def neighbours(self) -> List['Location']:
        neighbourList: List[Location] = []
        if self.x > 0: neighbourList.append(Location(self.x - 1, self.y))
        if self.x < self.max_x: neighbourList.append(Location(self.x + 1, self.y))
        if self.y > 0: neighbourList.append(Location(self.x, self.y - 1))
        if self.y < self.max_y: neighbourList.append(Location(self.x, self.y + 1))
        return neighbourList

    def get_neighbours_at_distance(self, distance: int, shuffle: bool = True, boundary_check: bool = True) -> List['Location']:
        """Gets all neighbours at a specific distance to the current location (manhattan distance).

        Args:
            distance (int): The proximity to the current location.
            shuffle (bool, optional): Whether to shuffle the neighbours. Defaults to True.
            boundary_check (bool, optional): Whether to check if the neighbours are within the boundaries of the world. Defaults to True.

        Returns:
            List[Location]: The neighbours at the specific distance to the current location.
        """
        neighbours = []
        for x in range(self.x - distance, self.x + distance + 1):
            for y in range(self.y - distance, self.y + distance + 1):
                if manhattan_distance(start=(self.x, self.y), end=(x, y)) == distance:
                    neighbour = Location(x, y)
                    if not boundary_check or (0 <= x <= self.max_x and 0 <= y <= self.max_y):
                        neighbours.append(neighbour)
        if shuffle:
            random.shuffle(neighbours)
        return neighbours

    def get_neighbour_by_orientation_and_action(self, orientation: Orientation, action: Action) -> 'Location':
        """Gets the location of the neighbour in the given orientation after taking the given action.

        Args:
            orientation (Orientation): The orientation of the agent.
            action (Action): The action taken by the agent.

        Returns:
            Location: The location of the neighbour in the given orientation after taking the given action.
        """
        new_location = copy.deepcopy(self)
        match action:
            case Action.LEFT:
                new_location = new_location.get_neighbour_location_on_left(orientation)
            case Action.RIGHT:
                new_location = new_location.get_neighbour_location_on_right(orientation)
            case Action.FORWARD:
                new_location.forward(orientation)
        return new_location

    def get_neighbour_location_on_left(self, orientation: Orientation) -> 'Location':
        """Gets the location of the neighbour on the left of the current location.

        Args:
            orientation (Orientation): The orientation of the agent.

        Returns:
            Location: The location of the neighbour on the left of the current location.
        """
        new_location = copy.deepcopy(self)
        match orientation:
            case Orientation.W:
                new_location.forward(Orientation.S)
            case Orientation.E:
                new_location.forward(Orientation.N)
            case Orientation.N:
                new_location.forward(Orientation.W)
            case Orientation.S:
                new_location.forward(Orientation.E)
        return new_location

    def get_neighbour_location_on_right(self, orientation: Orientation) -> 'Location':
        """Gets the location of the neighbour on the right of the current location.

        Args:
            orientation (Orientation): The orientation of the agent.

        Returns:
            Location: The location of the neighbour on the right of the current location.
        """
        new_location = copy.deepcopy(self)
        match orientation:
            case Orientation.W:
                new_location.forward(Orientation.N)
            case Orientation.E:
                new_location.forward(Orientation.S)
            case Orientation.N:
                new_location.forward(Orientation.E)
            case Orientation.S:
                new_location.forward(Orientation.W)
        return new_location

    def is_location(self, location: Union['Location', Tuple[int, int]]) -> bool:
        """Checks if the current location is the same as the given location.

        Args:
            location (Union[Location, Tuple[int, int]]): The location to compare to.

        Returns:
            bool: True if the locations are the same, False otherwise.
        """
        if isinstance(location, Location):
            return self.__eq__(location)
        else:
            return self.tuple == location

    def at_left_edge(self) -> bool:
        return self.x == 0

    def at_right_edge(self) -> bool:
        return self.x == self.max_x

    def at_top_edge(self) -> bool:
        return self.y == self.max_y

    def at_bottom_edge(self) -> bool:
        return self.y == 0

    def forward(self, orientation) -> bool:
        bump = False
        match orientation:
            case Orientation.W:
                if self.at_left_edge():
                    bump = True
                else:
                    self.x = self.x - 1
            case Orientation.E:
                if self.at_right_edge():
                    bump = True
                else:
                    self.x = self.x + 1
            case Orientation.N:
                if self.at_top_edge():
                    bump = True
                else:
                    self.y = self.y + 1
            case Orientation.S:
                if self.at_bottom_edge():
                    bump = True
                else:
                    self.y = self.y - 1
        return bump

    def get_forward_location(self, orientation: Orientation) -> 'Location':
        """Gets the location in front of the current location.

        Args:
            orientation (Orientation): The orientation of the agent.

        Returns:
            Location: The location in front of the current location.
        """
        new_location = copy.deepcopy(self)
        new_location.forward(orientation)
        return new_location

    def set_to(self, location: 'Location'):
        self.x = location.x
        self.y = location.y

    def random(self, exclude_point: Tuple[int, int] = None) -> 'Location':
        """Generates a random location.

        Args:
            exclude_point (Tuple[int, int], optional): A coordinate to exclude. Defaults to None.

        Returns:
            Location: The random location.
        """
        while True:
            self.x = random.randint(0, self.max_x)
            self.y = random.randint(0, self.max_y)
            if self.tuple != exclude_point:
                return self


class Environment:
    wumpus_location: Location
    wumpus_alive: bool
    has_wumpus: bool
    agent_location: Location
    agent_orientation: Orientation
    agent_has_arrow: bool
    agent_has_gold: bool
    game_over: bool
    gold_location: Location
    pit_locations: List[Location]
    time_step: int

    def __init__(
            self,
            world_size: Tuple[int, int] = (4, 4),
            agent_start_location: Tuple[int, int] = (0, 0),
            agent_escape_location: Tuple[int, int] = (0, 0),
            agent_start_orientation: Orientation = Orientation.E,
            agent_has_arrow: bool = True,
            agent_has_gold: bool = False,
            pit_prob: float = 0.2,
            allow_climb_without_gold: bool = False,
            has_wumpus: bool = True,
    ):
        """Initializes the environment.

        Args:
            world_size (Tuple[int, int], optional): The size of the world. Defaults to (4, 4).
            agent_start_location (Tuple[int, int], optional): The starting location of the agent. Defaults to (0, 0).
            agent_start_orientation (Orientation, optional): The starting orientation of the agent. Defaults to Orientation.E.
            agent_escape_location (Tuple[int, int], optional): The escape location of the agent. Defaults to (0, 0).
            agent_has_arrow (bool, optional): Whether the agent has an arrow. Defaults to True.
            agent_has_gold (bool, optional): Whether the agent has gold. Defaults to False.
            pit_prob (float, optional): The probability of a pit being in a location. Defaults to 0.2.
            allow_climb_without_gold (bool, optional): Whether the agent can climb without gold. Defaults to False.
            has_wumpus (bool, optional): Whether the world has a wumpus. Defaults to True.
        """
        self.world_size = world_size
        self.agent_location = Location(agent_start_location[0], agent_start_location[1], world_size=world_size)
        self.agent_orientation = agent_start_orientation
        self.agent_escape_location = Location(agent_escape_location[0], agent_escape_location[1], world_size=world_size)
        self.agent_has_arrow = agent_has_arrow
        self.agent_has_gold = agent_has_gold
        self.pit_prob = pit_prob
        self.allow_climb_without_gold = allow_climb_without_gold
        self.has_wumpus = has_wumpus

        self.make_wumpus(has_wumpus)
        self.make_gold()
        self.make_pits(pit_prob)
        self.game_over = False
        self.time_step = 0

    def get_initial_percept(self) -> Percept:
        """Gets the percept for the initial time step.

        Returns:
            Percept: The percept for the current time step.
        """
        return Percept(self.time_step, False, self.is_breeze(), self.is_stench(), False, False, self.agent_has_gold, 0, False)

    def make_wumpus(self, has_wumpus: bool):
        self.wumpus_location = Location(world_size=self.world_size).random(exclude_point=self.agent_location.tuple)
        self.wumpus_alive = has_wumpus

    def make_gold(self):
        self.gold_location = Location(world_size=self.world_size).random(exclude_point=self.agent_location.tuple)

    def make_pits(self, pit_prob: float):
        self.pit_locations = []
        for i in range(self.world_size[0]):
            for j in range(self.world_size[1]):
                if (i, j) != self.agent_location.tuple and random.random() < pit_prob:
                        self.pit_locations.append(Location(i, j))

    def is_pit_at(self, location: Location) -> bool:
        return any(pit_location.is_location(location) for pit_location in self.pit_locations)

    def is_pit_adjacent_to_agent(self) -> bool:
        for agent_neighbour in self.agent_location.neighbours():
            for pit_location in self.pit_locations:
                if agent_neighbour.is_location(pit_location):
                    return True
        return False

    def is_wumpus_adjacent_to_agent(self) -> bool:
        return self.has_wumpus and any(
            self.wumpus_location.is_location(neighbour) for neighbour in self.agent_location.neighbours())

    def is_agent_at_hazard(self) -> bool:
        return self.is_pit_at(self.agent_location) or (self.is_wumpus_at(self.agent_location) and self.wumpus_alive)

    def is_wumpus_at(self, location: Location) -> bool:
        return self.has_wumpus and self.wumpus_location.is_location(location)

    def is_agent_at(self, location: Location) -> bool:
        return self.agent_location.is_location(location)

    def is_gold_at(self, location: Location) -> bool:
        return self.gold_location.is_location(location)

    def is_glitter(self) -> bool:
        return self.is_gold_at(self.agent_location)

    def is_breeze(self) -> bool:
        return self.is_pit_adjacent_to_agent() or self.is_pit_at(self.agent_location)

    def is_stench(self) -> bool:
        return self.is_wumpus_adjacent_to_agent() or self.is_wumpus_at(self.agent_location)

    def wumpus_in_line_of_fire(self) -> bool:
        match self.agent_orientation:
            case Orientation.E:
                return self.has_wumpus and self.agent_location.is_left_of(self.wumpus_location)
            case Orientation.S:
                return self.has_wumpus and self.agent_location.is_above(self.wumpus_location)
            case Orientation.W:
                return self.has_wumpus and self.agent_location.is_right_of(self.wumpus_location)
            case Orientation.N:
                return self.has_wumpus and self.agent_location.is_below(self.wumpus_location)

    def kill_attempt(self) -> bool:
        if not (self.has_wumpus and self.wumpus_alive): return False
        scream = self.wumpus_in_line_of_fire()
        self.wumpus_alive = not scream
        return scream

    def step(self, action: Action) -> Percept:
        special_reward = 0
        bump = False
        scream = False
        # if self.time_step == 999:
        #  self.game_over = True
        if self.game_over:
            reward = 0
        else:
            match action:
                case Action.LEFT:
                    self.agent_orientation = self.agent_orientation.turn_left()
                case Action.RIGHT:
                    self.agent_orientation = self.agent_orientation.turn_right()
                case Action.FORWARD:
                    bump = self.agent_location.forward(self.agent_orientation)
                    if self.agent_has_gold: self.gold_location.set_to(self.agent_location)
                    if self.is_agent_at_hazard():
                        special_reward = -1000
                        self.game_over = True
                case Action.GRAB:
                    if self.agent_location.is_location(self.gold_location):
                        self.agent_has_gold = True
                case Action.SHOOT:
                    if self.agent_has_arrow:
                        scream = self.kill_attempt()
                        special_reward = -10
                        self.agent_has_arrow = False
                case Action.CLIMB:
                    if self.agent_location.is_location(self.agent_escape_location):
                        if self.agent_has_gold:
                            special_reward = 1000
                        if self.allow_climb_without_gold or self.agent_has_gold:
                            self.game_over = True
            reward = -1 + special_reward

        breeze = self.is_breeze()
        stench = self.is_stench()
        glitter = self.is_glitter()
        self.time_step = self.time_step + 1
        return Percept(self.time_step, bump, breeze, stench, scream, glitter, self.agent_has_gold, reward, self.game_over)

    def visualize(
            self,
            visualize_paths: Tuple[Tuple[Union[Set[Tuple[int, int]], List[Tuple[int, int]]], str]] = (),
    ):
        """Visualizes the environment.

        Args:
            visualize_paths (Tuple[Tuple[Union[Set[Tuple[int, int]], List[Tuple[int, int]]], str]], optional): A tuple of paths to visualize and their symbols. Defaults to ().
        """
        for y in range(self.world_size[1] - 1, -1, -1):
            line = '|'
            for x in range(0, self.world_size[0]):
                loc = Location(x, y)
                cell_symbols = [' '] * 4
                for path, symbol in visualize_paths:
                    if path and loc.tuple in path: cell_symbols[0] = symbol
                if self.is_agent_at(loc): cell_symbols[0] = self.agent_orientation.symbol()
                if self.is_pit_at(loc): cell_symbols[1] = 'P'
                if self.has_wumpus and self.is_wumpus_at(loc):
                    if self.wumpus_alive:
                        cell_symbols[2] = 'W'
                    else:
                        cell_symbols[2] = 'm'
                if self.is_gold_at(loc): cell_symbols[3] = 'G'
                for char in cell_symbols: line += char
                line += '|'
            print(line)


@dataclass
class GameConfig:
    world_size: Optional[Tuple[int, int]] = (4, 4)
    agent_start_location: Optional[Tuple[int, int]] = (0, 0)
    agent_escape_location: Optional[Tuple[int, int]] = (0, 0)
    agent_start_orientation: Optional[Orientation] = Orientation.E
    agent_has_arrow: Optional[bool] = True
    agent_has_gold: Optional[bool] = False
    pit_prob: Optional[float] = 0.2
    allow_climb_without_gold: Optional[bool] = False
    has_wumpus: Optional[bool] = False

    def __str__(self) -> str:
        """Gets the string representation of the game configuration, in multi-line format.

        Returns:
            str: The string representation of the game configuration
        """
        return f'World Size: {self.world_size}\n' \
               f'Agent Start Location: {self.agent_start_location}\n' \
               f'Agent Escape Location: {self.agent_escape_location}\n' \
               f'Agent Start Orientation: {self.agent_start_orientation}\n' \
               f'Agent Has Arrow: {self.agent_has_arrow}\n' \
               f'Agent Has Gold: {self.agent_has_gold}\n' \
               f'Pit Probability: {self.pit_prob}\n' \
               f'Allow Climb Without Gold: {self.allow_climb_without_gold}\n' \
               f'Has Wumpus: {self.has_wumpus}'

if __name__ == "__main__":
    game_config = GameConfig(
        world_size=(4, 4),
        agent_start_location=(0, 0),
        agent_start_orientation=Orientation.E,
        agent_escape_location=(0, 0),
        agent_has_arrow=False,
        agent_has_gold=False,
        pit_prob=0,   # 0.2
        allow_climb_without_gold=False,
        has_wumpus=False,
    )
    agent = Agent(game_config=game_config)
    agent.run()