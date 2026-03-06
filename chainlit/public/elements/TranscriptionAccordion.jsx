import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";

export default function TranscriptionAccordion({ props }) {
  const audioSrc = props.audioBase64
    ? `data:${props.audioMime || "audio/wav"};base64,${props.audioBase64}`
    : null;

  return (
    <Accordion type="single" collapsible className="w-full">
      <AccordionItem value="transcription">
        <AccordionTrigger>
          📝 {props.label ? `${props.label} — ` : ""}Ver transcrição completa ({props.characterCount} caracteres)
        </AccordionTrigger>
        <AccordionContent>
          {audioSrc && (
            <audio controls className="w-full mb-2">
              <source src={audioSrc} type={props.audioMime || "audio/wav"} />
            </audio>
          )}
          <div className="whitespace-pre-wrap text-sm">
            {props.transcription}
          </div>
        </AccordionContent>
      </AccordionItem>
    </Accordion>
  );
}
