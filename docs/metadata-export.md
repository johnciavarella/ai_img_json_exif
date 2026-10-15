# Metadata Export & EXIF / IPTC / XMP Mapping

The application provides non-destructive metadata export. When you export an image or batch, your original source files are **never modified**. The tool creates a new copy of the full-resolution file and writes the model prediction results into standard metadata fields using `exiftool`.

## Field Mapping

Any JSON response (from an LLM, vision model, or workflow) is recursively inspected for known keys:

| JSON Key | Target Metadata Tags |
|---|---|
| `description`, `caption`, `summary` | `IFD0:ImageDescription`<br>`XMP-dc:Description`<br>`IPTC:Caption-Abstract` |
| `title`, `headline` | `XMP-dc:Title`<br>`IPTC:ObjectName`<br>`XMP-photoshop:Headline`<br>`IPTC:Headline` |
| `tags`, `keywords`, `labels`, `objects`, `category` | `XMP-dc:Subject`<br>`IPTC:Keywords` |
| `rating` (integer 0–5) | `XMP-xmp:Rating` |
| `artist`, `creator`, `author` | `IFD0:Artist`<br>`XMP-dc:Creator` |
| `copyright` | `IFD0:Copyright`<br>`XMP-dc:Rights` |
| `city`, `state`, `country`, `location` | `XMP-photoshop:*`<br>`XMP-iptcCore:Location` |
| `event` | `XMP-iptcExt:Event` |
| `people`, `persons` | `XMP-iptcExt:PersonInImage` |

## Safety & Non-Destructive Guarantees

1. **Camera Tags Protected**: Hardware and capture metadata (Camera Make, Model, Lens, Shutter Speed, Aperture, ISO, Original Capture Date, GPS coordinates) are **never** modified or overwritten.
2. **Preserving Existing Tags**: By default, existing metadata is preserved. If a photo already has a caption or keywords, new tags are appended or existing ones kept, unless you explicitly enable **Overwrite existing metadata**.
3. **Full JSON in UserComment**: The complete, unparsed JSON output is stored in the `UserComment` field so that custom application logic can always recover the raw model predictions.

