# CraftWorld

### Crafting Rules
- The agent can get raw materials: `wood`, `grass`, `iron`
- The agent can go to the `toolshed` with stored raw materials to craft new objects:
    - make `plank` from `wood`
    - make `rope` from `grass`
    - make `axe` from `stick` + `iron`
    - make `bow` from `rope` + `stick`
- The agent can go to the `workbench` with stored raw materials to craft new objects:
    - make `stick` from `wood`
    - make `saw` from `iron`
    - make `bed` from `plank` + `grass`
    - make `shears` from `stick` + `iron`
- The agent can go to the `factory` with stored raw materials to craft new objects:
    - make `cloth` from `grass`
    - make `bridge` from `iron` + `wood`
    - make `goldware` from `gold`
    - make `ring` from `gem`
- The agent can only get `gold` after it has crafted a `bridge`
- The agent can only get `gem` after it has crafted an `axe`

### Event Detectors
- 'a' is emitted when agent goes to get wood
- 'b' is emitted when agent goes to the toolshed
- 'c' is emitted when agent goes to the workbench
- 'd' is emitted when agent goes to get grass
- 'e' is emitted when agent goes to the factory
- 'f' is emitted when agent goes to get iron
- 'g' is emitted when agent goes to get gold
- 'h' is emitted when agent goes to get gem

### Landmarks
The 'reward_machines' folder contains RM to reach landmarks

- have-axe
- have-bed
- have-bridge
- have-cloth
- have-gem
- have-gold
- have-grass
- have-iron
- have-plank
- have-rope
- have-shears
- have-stick
- have-wood