import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";

export default function TranscriptionAccordion() {
  const p = (typeof props !== "undefined" && props) || {};
  const audioSrc = p.audioBase64
    ? `data:${p.audioMime || "audio/wav"};base64,${p.audioBase64}`
    : null;

  return (
    <Accordion type="single" collapsible className="w-full">
      <AccordionItem value="transcription">
        <AccordionTrigger>
          📝 {p.label ? `${p.label} — ` : ""}Ver transcrição completa ({p.characterCount} caracteres)
        </AccordionTrigger>
        <AccordionContent>
          {audioSrc && (
            <audio controls className="w-full mb-2">
              <source src={audioSrc} type={p.audioMime || "audio/wav"} />
            </audio>
          )}
          <div className="whitespace-pre-wrap text-sm">
            {p.transcription}
          </div>
        </AccordionContent>
      </AccordionItem>
    </Accordion>
  );
}
