import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";

export default function TranscriptionAccordion() {
  return (
    <Accordion type="single" collapsible className="w-full">
      <AccordionItem value="transcription">
        <AccordionTrigger>
          📝 Ver transcrição completa ({props.characterCount} caracteres)
        </AccordionTrigger>
        <AccordionContent>
          <div className="whitespace-pre-wrap text-sm">
            {props.transcription}
          </div>
        </AccordionContent>
      </AccordionItem>
    </Accordion>
  );
}
