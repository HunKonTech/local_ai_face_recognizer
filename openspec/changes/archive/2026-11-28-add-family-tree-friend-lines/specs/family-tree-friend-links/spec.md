# Spec Delta

## Purpose

Shows friends and acquaintances recorded in family codes on the family tree, connected to their anchor person with dashed lines, switchable for the whole chart or for chosen persons.

## ADDED Requirements

### Requirement: Derive friend links from family codes
A person whose family code contains a friend code SHALL be linked as a friend to the person whose family code equals that code without the friend marker. Multi-codes SHALL produce one link per listed code. Ranges SHALL produce one link per expanded code. Codes without a matching anchor SHALL produce no link. The active code scheme's friend marker SHALL be used.

#### Scenario: Single friend code
- **WHEN** person F has family code `C81B` and person A has family code `C81`
- **THEN** F is a friend of A

#### Scenario: Shared friend
- **WHEN** person F has `C81B,C82B` and persons A (`C81`) and B (`C82`) exist
- **THEN** F is a friend of both A and B

#### Scenario: Range
- **WHEN** person F has `C[1-3]B` and only `C1` and `C3` exist
- **THEN** F is a friend of the persons with `C1` and `C3`

#### Scenario: No anchor
- **WHEN** person F has `C99B` and nobody has `C99`
- **THEN** F has no friend link

### Requirement: Friend display modes
The family tree SHALL support friend display modes off, everyone and selected persons. In off mode no friend link is drawn. In everyone mode friend links of every visible anchor are drawn. In selected mode only friend links of the chosen anchors are drawn.

#### Scenario: Off by default
- **WHEN** a new view is created
- **THEN** friend mode is off and no friend is drawn

#### Scenario: Selected person
- **WHEN** mode is selected persons and only A is selected, and A and B both have friends
- **THEN** only A's friends are drawn

### Requirement: Dashed friend lines
A drawn friend link SHALL appear as a dashed line between the friend's box and the anchor's box, visually distinct from spouse and parent-child lines.

#### Scenario: Line style
- **WHEN** friend F of A is drawn
- **THEN** F and A are connected by a dashed line

### Requirement: Placement of friends
A friend already on the chart SHALL be connected at their existing position. A friend not on the chart SHALL be shown once, next to their first visible anchor in that anchor's generation row, with dashed lines to every visible anchor whose friends are drawn.

#### Scenario: Friend is a family member
- **WHEN** friend F of A is already on the chart as a family member
- **THEN** no second F box appears and the dashed line connects the existing boxes

#### Scenario: Shared friend not in the family
- **WHEN** F is a friend of A and B, F is not otherwise on the chart, and both A's and B's friends are drawn
- **THEN** one F box appears next to A with dashed lines to A and B

### Requirement: Friends in saved views
In Tree Studio the friend mode and the selected anchors SHALL be saved with the view and restored on load. A person's context menu SHALL toggle whether that person's friends are shown. A friend box SHALL be hideable like any person.

#### Scenario: Save and reload
- **WHEN** a view in selected mode with anchor A is saved and loaded again
- **THEN** A's friends are drawn and nobody else's

#### Scenario: Hide a friend
- **WHEN** the user hides friend F
- **THEN** F's box and dashed lines disappear and F appears in the hidden-persons list

### Requirement: Toggle in Face-Local's tree tab
Face-Local's family tree tab SHALL offer a toggle between off and everyone, remembered between runs.

#### Scenario: Toggle on
- **WHEN** the user turns on friends in the family tree tab
- **THEN** friends of every visible person are drawn with dashed lines
- **AND** the setting is still on after restarting Face-Local
