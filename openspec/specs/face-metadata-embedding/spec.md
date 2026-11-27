# face-metadata-embedding Specification

## Purpose

Defines how Face-Local writes person and face data into image files and reads it back, so that the data is visible in common OS and photo tools, existing metadata is preserved, and the user always knows where the data ended up.

## Requirements

### Requirement: Person names visible in Windows file properties
When face metadata is embedded into a JPEG, the system SHALL write the names of the known persons into fields that Windows Explorer shows as "Comments" and "Tags", encoded so that non-ASCII characters (e.g. Hungarian accents) display correctly. Unknown faces SHALL NOT contribute a name.

#### Scenario: Known persons appear in Explorer
- **WHEN** a JPEG with faces assigned to "Szilvay Géza" and "Kiss Anna" is exported with names included
- **THEN** Windows Explorer's Comments property of the file lists both names with correct accents
- **AND** its Tags property contains each name as a separate tag

#### Scenario: Names excluded by the user
- **WHEN** the export runs with person names disabled
- **THEN** no person name is written to the Comments or Tags fields

#### Scenario: Only unknown faces
- **WHEN** an image has only unassigned faces
- **THEN** no name is written to the Comments or Tags fields

### Requirement: Named face regions in standard XMP
When face metadata is embedded and face boxes are included, the system SHALL write each known person's face rectangle as a named region in both the Microsoft Photo region format and the MWG region format, using coordinates normalized to the image dimensions.

#### Scenario: People shown by Windows
- **WHEN** a JPEG with an assigned face is exported with names and boxes included
- **THEN** Windows Explorer's People property lists that person

#### Scenario: Region readable by MWG tools
- **WHEN** the exported file is read by an MWG-compliant tool
- **THEN** it finds one face region per known person, named after the person, whose rectangle covers the face shown in the app

#### Scenario: Boxes excluded by the user
- **WHEN** the export runs with face boxes disabled
- **THEN** no region is written

### Requirement: Regions match the displayed image orientation
Region rectangles SHALL cover the face as it appears in the upright image shown by the app and by viewers that honor the EXIF orientation, for every EXIF orientation value.

#### Scenario: Rotated phone photo
- **WHEN** a JPEG with EXIF orientation 6 (rotated 90 degrees) is exported
- **THEN** a viewer that applies the orientation draws each region over the same face the app shows

### Requirement: App round-trip data preserved
The system SHALL embed the full versioned `facelocal.faces.v1` document in JPEGs both in the EXIF UserComment field and in XMP, with identical content (PNG: XMP only), and SHALL read back person data from files written by this or earlier versions.

#### Scenario: JSON in both locations
- **WHEN** a JPEG is exported
- **THEN** an EXIF reader finds the `facelocal.faces.v1` JSON in UserComment, with accented names readable
- **AND** an XMP reader finds the same document in XMP

#### Scenario: Read back new file
- **WHEN** a file exported by this version is read by the app
- **THEN** the app recovers the same `facelocal.faces.v1` document that was written

#### Scenario: Read back legacy file
- **WHEN** a file exported by the previous version (JSON in EXIF UserComment with the ASCII character code, or in the `facelocal:data` XMP property) is read
- **THEN** the app recovers its `facelocal.faces.v1` document

### Requirement: No mis-encoded text in ASCII-only fields
The system SHALL NOT write non-ASCII text into EXIF fields whose standard encoding is ASCII, and text stored in EXIF UserComment SHALL declare a character code that matches its bytes.

#### Scenario: Accented name
- **WHEN** a file with a person named "Szilvay Géza" is exported
- **THEN** no EXIF field displays the name as "Szilvay G?za" or other mojibake in Windows Explorer or an EXIF reader

### Requirement: Re-export replaces, never duplicates
Repeated exports of the same image SHALL leave exactly one set of Face-Local data (one JSON document, one set of regions, one set of name tags) reflecting the latest export, and SHALL remove names or regions of persons no longer assigned.

#### Scenario: Person removed between exports
- **WHEN** an image is exported with persons A and B, then B is unassigned and the image is exported again
- **THEN** the file's Comments, Tags and regions contain only A

### Requirement: Existing metadata preserved
Embedding SHALL preserve the file's existing EXIF data (including GPS, capture date and orientation), its ICC color profile, and XMP properties not owned by Face-Local. The person comment, tag and region fields the system writes are owned by it and are replaced as a whole on each export.

#### Scenario: GPS, date and color profile survive
- **WHEN** a JPEG with GPS, DateTimeOriginal, orientation and an embedded ICC profile is exported
- **THEN** after the export all four are still present with unchanged values

#### Scenario: Foreign XMP survives
- **WHEN** a JPEG containing an unrelated XMP property (e.g. a rating) is exported
- **THEN** that property is still present afterwards

### Requirement: One file rewrite per save action
When one user action saves several kinds of metadata (GPS, date, persons) for an image, the system SHALL apply them in a single write of that image file.

#### Scenario: Save all metadata from the image browser
- **WHEN** the user saves GPS, date and persons for selected images
- **THEN** each image file is replaced at most once by that action

### Requirement: Locked files reported, data not lost
When an image cannot be updated because another program holds it open, the system SHALL keep the data in a sidecar JSON file, SHALL report that image as locked (distinct from a deliberate sidecar export and from a failure), and the summary SHALL list the locked files so the user can retry after closing the other program.

#### Scenario: File held open by a viewer
- **WHEN** a JPEG is open in another program that blocks writing, and the user exports its face metadata
- **THEN** the result for that image is "locked"
- **AND** a sidecar JSON with the data exists next to the image
- **AND** the export summary shows the locked count and the file name

#### Scenario: Retry after the lock is gone
- **WHEN** the other program is closed and the user exports the same image again
- **THEN** the data is embedded into the image and the result is an embedded mode

#### Scenario: Short transient lock
- **WHEN** the file is locked only briefly (e.g. by the indexer right after a write)
- **THEN** the export still embeds the data without user action

### Requirement: Image content not damaged
A failed or interrupted write SHALL leave the original image file intact and readable, and no temporary files SHALL remain next to the image afterwards.

#### Scenario: Write fails mid-way
- **WHEN** replacing the image fails for any reason
- **THEN** the original file is byte-identical to before the export and no `*.facelocal.tmp` file remains
