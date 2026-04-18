"""
Knowledge Graph Triples Extractor Helper

We use REBEL from hugging face (https://huggingface.co/Babelscape/rebel-large)

REBEL is a seq2seq model. We feed it the raw text and it outputs
a special markup string like:
<triplet> Fed <subj> raised <rel> interest rates <obj>
We then parse that markup into clean tuples.
"""

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import re
from typing import List, Tuple
import spacy

nlp = spacy.load("en_core_web_sm")

# Triple is an alias for 3 variable tuple (head enitity,relation,tail entity)
Triple = Tuple[str, str, str]


class KGExtractor:

    # REBEL is ~1.6GB and has context window of 512 tokens
    model_name = "Babelscape/rebel-large"
    MAX_INPUT_TOKENS = 480  # leave room for REBEL's special tokens
    MAX_NEW_TOKENS = 256

    def __init__(self, device: str | None = "cpu"):
        """
        Args:
            device: "cuda", "cpu", or None (auto-detect)
        """

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Loading REBEL extractor on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.model.eval()
        self.model.to(self.device)
        print("REBEL loaded")

    def extract(
        self,
        text: str,
    ):
        """
        Main function to extract triples from a single string.

        Args:
            text: Input text. Can be a sentence or a paragraph. Long texts are chunked automatically.
            max_triples: Upper cap on how many triples to return.
            option:
                'str' - output is list[str]
                'dict' - output is list[dict]
                 None - output is list[Triple]
            deduplicate:
                whether to dedeuplicate the triples or not
                It is expsensive hence the default is False
        """

        # REBEL has a 512-token limit, so we chunk long texts
        chunks = self._chunk_text(text)

        all_triples: List[Triple] = []
        for chunk in chunks:
            raw_output = self._run_model(chunk)
            triples = self._parse_rebel_output(raw_output)
            all_triples.extend(triples)

        return all_triples

    def extract_chunk_batch(self, text: str, batch_size: int = 8) -> List[Triple]:
        chunks = self._chunk_by_sentences(text)
        triples: dict[Triple, None] = {} # Use dict instead of set for preserving insertion order

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            for raw in self._run_model_batch(batch):
                for t in self._parse_rebel_output(raw):
                    triples[t] = None

        return list(triples)

    def _chunk_by_sentences(self, text, max_tokens=512):
        doc = nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]

        chunks, current_chunk = [], []
        current_len = 0

        for sent in sentences:
            sent_len = len(self.tokenizer(sent, add_special_tokens=False)["input_ids"])
            if current_len + sent_len > max_tokens:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [sent]
                current_len = sent_len
            else:
                current_chunk.append(sent)
                current_len += sent_len

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def extract_batch(
        self,
        texts: List[str],
    ) -> List[List[Triple]]:
        """
        Extract triples from a list of text (one list of triples per text).

        Args:
            texts: list of documents
            max_triples: maximum triples, default = 100
        """
        return [self.extract(t) for t in texts]

    def _run_model(self, text: str) -> str:
        """
        Run REBEL on a single chunk, return raw output string
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=256,
            )

        # Decode the first output token ID back to a string
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=False)

    def _run_model_batch(self, chunks: List[str]) -> List[str]:
        """
        Run REBEL on multiple chunks in a single forward pass.
        """
        inputs = self.tokenizer(
            chunks,  # list instead of single string
            return_tensors="pt",
            max_length=self.MAX_INPUT_TOKENS,
            truncation=True,
            padding=True,  # pads shorter chunks to match the longest
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.MAX_NEW_TOKENS,
            )

        # output_ids shape: (num_chunks, seq_len) — decode each row
        return [
            self.tokenizer.decode(ids, skip_special_tokens=False) for ids in output_ids
        ]

    def _chunk_text(
        self, text: str, chunk_size: int = 512, overlap: int = 50
    ) -> List[str]:
        truncated_ids = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=4096,
        )["input_ids"]

        chunks = []
        step = chunk_size - overlap
        for start in range(0, len(truncated_ids), step):
            chunk_ids = truncated_ids[start : start + chunk_size]
            chunks.append(self.tokenizer.decode(chunk_ids, skip_special_tokens=True))

        return chunks

    def _parse_rebel_output(self, text: str) -> List[Triple]:
        """
        Function taken from hugging face https://huggingface.co/Babelscape/rebel-large
        and modified
        Args:
            text: decoded text from REBEL
            option:
                'str' - output is list[str]
                'dict' - output is list[dict]
                None - output is list[Triple]
        """

        triplets, seen = [], set()
        for chunk in text.split("<triplet>"):
            chunk = chunk.strip()
            if not chunk:
                continue

            parts = re.split(r"<subj>|<obj>", chunk)

            if len(parts) < 3:
                continue

            head, tail, rel = parts[0].strip(), parts[1].strip(), parts[2].strip()

            head = re.sub(r"</s>|<s>|<pad>", "", head).strip()
            rel = re.sub(r"</s>|<s>|<pad>", "", rel).strip()
            tail = re.sub(r"</s>|<s>|<pad>", "", tail).strip()

            if head and rel and tail and head.lower() != tail.lower():
                key = (head, rel, tail)
                if key not in seen:
                    seen.add(key)
                    triplets.append((head, rel, tail))

        return triplets


# Example use case
if __name__ == "__main__":
    extractor = KGExtractor()

    article1 = """
    The Federal Reserve raised interest rates by 0.25 percentage points on Wednesday,
    the tenth consecutive increase, as it continued its battle against inflation.
    Fed Chair Jerome Powell said the committee remained committed to bringing inflation
    down to its 2% target. Markets fell sharply following the announcement,
    with the S&P 500 dropping 1.8%.
    """

    article2 = """
    Javokhir Sindarov missed a clear opportunity to secure victory this afternoon against Matthias Bluebaum, allowing Anish Giri to narrow the gap 
    after defeating Fabiano Caruana with the black pieces. With five rounds remaining, Sindarov still leads by 1.5 points.Vaishali and Zhu Jiner 
    share the lead at the FIDE Women's Candidates with 5.5/9, following victories over Divya Deshmukh and Kateryna Lagno, respectively. 
    Meanwhile, Anna Muzychuk squandered a highly promising endgame, slipping behind the leaders.
    Let's take a closer look at how the afternoon unfolded. The ceremonial opening move was played by Paris Klerides, General Secretary of 
    the Cyprus Chess Federation and FIDE Delegate for Cyprus, who made the symbolic 1.e4 on behalf of Matthias Bluebaum. However, 
    Bluebaum opted for 1.d4 instead. Javokhir Sindarov replied with a very rare line 
    the Harrwitz Attack in the Queen's Gambit Decline
    """

    article3 = """
    a recent systematic analysis showed that in 2011 , 314 ( 296 - 331 ) million children younger than 5 years were mildly , moderately or severely stunted and 258 ( 240 - 274 ) million were mildly , moderately or severely underweight in the developing countries . \n in iran a study among 752 high school girls in sistan and baluchestan showed prevalence of 16.2% , 8.6% and 1.5% , for underweight , overweight and obesity , respectively . \n the prevalence of malnutrition among elementary school aged children in tehran varied from 6% to 16% . \n anthropometric study of elementary school students in shiraz revealed that 16% of them suffer from malnutrition and low body weight . \n snack should have 300 - 400 kcal energy and could provide 5 - 10 g of protein / day . nowadays , school nutrition programs are running as the national programs , world - wide . national school lunch program in the united states \n there are also some reports regarding school feeding programs in developing countries . in vietnam , \n school base program showed an improvement in nutrient intakes . in iran a national free food program ( nffp ) \n is implemented in elementary schools of deprived areas to cover all poor students . however , this program is not conducted in slums and poor areas of the big cities so many malnourished children with low socio - economic situation are not covered by nffp . although the rate of poverty in areas known as deprived is higher than other areas , many students in deprived areas are not actually poor and can afford food . \n hence , nutritional value of the nffp is lower than the scientific recommended snacks for this age group . \n furthermore , lack of variety of food packages has decreased the tendency of children toward nffp . on the other hand , \n the most important one is ministry of education ( moe ) of iran , which is responsible for selecting and providing the packages for targeted schools . \n the ministry of health ( moh ) is supervising the health situation of students and their health needs . \n welfare organizations , along with charities , have the indirect effect on nutritional status of students by financial support of their family . \n provincial governors have also the role of coordinating and supervising all activities of these organizations . \n parent - teacher association is a community - based institution that participates in school 's policy such as nffp . \n in addition to these organizations , nutritional literacy of students , their parents and teachers , is a very important issue , which could affect nutritional status of school age children . \n therefore , the present study was conducted with the aim of improving the nffp , so that by its resources all poor children will be covered even in big cities . \n moreover , all food packages were replaced by nutritious and diverse packages that were accessible for non - poor children . according to the aim of this study and multiple factors that could affect the problem , \n public health advocacy has been chosen as the best strategy to deal with this issue . \n therefore , the present study determines the effects of nutrition intervention in an advocacy process model on the prevalence of underweight in school aged children in the poor area of shiraz , iran . \n this interventional study has been carried out between 2009 and 2010 in shiraz , iran . \n this survey was approved by the research committee of shiraz university of medical sciences . in coordination with education organization of fars province \n two elementary schools and one middle school in the third region of the urban area of shiraz were selected randomly . in those schools all \n students ( 2897 , 7 - 13 years old ) were screened based on their body mass index ( bmi ) by nutritionists . according to convenience method all \n students divided to two groups based on their economic situation ; family revenue and head of household 's job and nutrition situation ; the first group were poor and malnourished students and the other group were well nourished or well - off students . \n for this report , the children 's height and weight were entered into center for disease control and prevention ( cdc ) to calculate bmi and bmi - for - age z - scores based on cdc for diseases control and prevention and growth standards . \n the significance of the difference between proportions was calculated using two - tailed z - tests for independent proportions . for implementing the interventions , \n the advocacy process model weight was to the nearest 0.1 kg on a balance scale ( model # seca scale ) . \n standing height was measured to the nearest 0.1 cm with a wall - mounted stadiometer . \n advocacy group formation : this step was started with stakeholder analysis and identifying the stakeholders . \n the team was formed with representatives of all stakeholders include ; education organization , welfare organization , deputy for health of shiraz university , food and cosmetic product supervisory office and several non - governmental organizations and charities . situation analysis : this was carried out by use of existing data such as formal report of organizations , literature review and focus group with experts . \n the prevalence of malnutrition and its related factors among students was determined and weaknesses and strengths of the nffp were analyzed . \n accordingly , three sub - groups were established : research and evaluation , education and justification and executive group . designing the strategies : \n three strategies were identified ; education and justification campaign , nutritional intervention ( providing nutritious , safe and diverse snacks ) and networking . performing the interventions : interventions that were implementing in selected schools were providing a diverse and nutritious snack package along with nutrition education for both groups while the first group ( poor and malnourished students ) was utilized the package free of charge . \n education and justification intervention : regarding the literature review and expert opinion , an educational group affiliated with the advocacy team has prepared educational booklets about nutritional information for each level ( degree ) . \n accordingly , education of these booklets has been integrated into regular education of students and they educated and justified for better nutrition life - style . \n it leads the educational group to hold several meeting with the student 's parents to justify them about the project and its benefit for their children . \n after these meetings , parental desire for participation in the project illustrated the effectiveness of the justification meeting with them . \n for educate fifteen talk show programs in tv and radio , 12 published papers in the local newspaper , have implemented to mobilize the community and gain their support . \n healthy diet , the importance of breakfast and snack in adolescence , wrong food habits among school age children , role of the family to improve food habit of children were the main topics , in which media campaign has focused on . \n nutritional intervention : the snack basket of the students was replaced with traditional , nutritious and diverse foods . in general , the new snack package in average has provided 380 kcal energy , 15 g protein along with sufficient calcium and iron . \n low economic and malnourished children were supported by executive group affiliated with advocacy team and the rest of them prepare their snack by themselves . \n research and evaluation : in this step , the literacy and anthropometric indices ( bmi ) of students were assessed before and after the interventions . \n the reference for anthropometric measures was the world health organization / national center for health statistics ( who / nchs ) standards and the cut - offs were - two standard deviations ( sd ) from the mean . \n each student that was malnourished and poor has been taken into account for free food and nutritious snacks . \n demographic information , height , weight and knowledge of the students were measured by use of a validated and reliable ( cronbach 's alpha was 0.61 ) questionnaire . \n this project is granted by shiraz university of medical sciences , charities and welfare organization and education organization of fars province . \n statistical analyses were performed using the statistical package for the social sciences ( spss ) software , version 17.0 ( spss inc . , \n the results are expressed as mean  sd and proportions as appropriated . in order to determine the effective variables on the malnutrition status \n paired t test was used to compare the end values with baseline ones in each group . \n in this project , the who z - score cut - offs used were as follow : using bmi - for - age z - scores ; overweight : > + 1 sd , i.e. , z - score > 1 ( equivalent to bmi 25 kg / m ) , obesity : > + 2 sd ( equivalent to bmi 30 kg / m ) , thinness : < 2 sd and severe thinness : < 3 sd . \n this interventional study has been carried out between 2009 and 2010 in shiraz , iran . \n this survey was approved by the research committee of shiraz university of medical sciences . in coordination with education organization of fars province \n two elementary schools and one middle school in the third region of the urban area of shiraz were selected randomly . in those schools all \n students ( 2897 , 7 - 13 years old ) were screened based on their body mass index ( bmi ) by nutritionists . according to convenience method all \n students divided to two groups based on their economic situation ; family revenue and head of household 's job and nutrition situation ; the first group were poor and malnourished students and the other group were well nourished or well - off students . \n for this report , the children 's height and weight were entered into center for disease control and prevention ( cdc ) to calculate bmi and bmi - for - age z - scores based on cdc for diseases control and prevention and growth standards . \n the significance of the difference between proportions was calculated using two - tailed z - tests for independent proportions . for implementing the interventions , \n weight was to the nearest 0.1 kg on a balance scale ( model # seca scale ) . \n standing height was measured to the nearest 0.1 cm with a wall - mounted stadiometer . \n advocacy group formation : this step was started with stakeholder analysis and identifying the stakeholders . \n the team was formed with representatives of all stakeholders include ; education organization , welfare organization , deputy for health of shiraz university , food and cosmetic product supervisory office and several non - governmental organizations and charities . situation analysis : this was carried out by use of existing data such as formal report of organizations , literature review and focus group with experts . \n the prevalence of malnutrition and its related factors among students was determined and weaknesses and strengths of the nffp were analyzed . \n accordingly , three sub - groups were established : research and evaluation , education and justification and executive group . designing the strategies : \n three strategies were identified ; education and justification campaign , nutritional intervention ( providing nutritious , safe and diverse snacks ) and networking . performing the interventions : interventions that were implementing in selected schools were providing a diverse and nutritious snack package along with nutrition education for both groups while the first group ( poor and malnourished students ) was utilized the package free of charge . duration of intervention was 6 months . \n education and justification intervention : regarding the literature review and expert opinion , an educational group affiliated with the advocacy team has prepared educational booklets about nutritional information for each level ( degree ) . \n accordingly , education of these booklets has been integrated into regular education of students and they educated and justified for better nutrition life - style . obviously , student 's families had remarkable effect on children 's food habit . \n it leads the educational group to hold several meeting with the student 's parents to justify them about the project and its benefit for their children . \n after these meetings , parental desire for participation in the project illustrated the effectiveness of the justification meeting with them . \n educate fifteen talk show programs in tv and radio , 12 published papers in the local newspaper , have implemented to mobilize the community and gain their support . \n healthy diet , the importance of breakfast and snack in adolescence , wrong food habits among school age children , role of the family to improve food habit of children were the main topics , in which media campaign has focused on . \n nutritional intervention : the snack basket of the students was replaced with traditional , nutritious and diverse foods . in general , the new snack package in average has provided 380 kcal energy , 15 g protein along with sufficient calcium and iron . \n low economic and malnourished children were supported by executive group affiliated with advocacy team and the rest of them prepare their snack by themselves . \n research and evaluation : in this step , the literacy and anthropometric indices ( bmi ) of students were assessed before and after the interventions . \n the reference for anthropometric measures was the world health organization / national center for health statistics ( who / nchs ) standards and the cut - offs were - two standard deviations ( sd ) from the mean . \n each student that was malnourished and poor has been taken into account for free food and nutritious snacks . \n demographic information , height , weight and knowledge of the students were measured by use of a validated and reliable ( cronbach 's alpha was 0.61 ) questionnaire . \n this project is granted by shiraz university of medical sciences , charities and welfare organization and education organization of fars province . \n advocacy group formation : this step was started with stakeholder analysis and identifying the stakeholders . \n the team was formed with representatives of all stakeholders include ; education organization , welfare organization , deputy for health of shiraz university , food and cosmetic product supervisory office and several non - governmental organizations and charities . \n situation analysis : this was carried out by use of existing data such as formal report of organizations , literature review and focus group with experts . \n the prevalence of malnutrition and its related factors among students was determined and weaknesses and strengths of the nffp were analyzed . \n accordingly , three sub - groups were established : research and evaluation , education and justification and executive group . \n designing the strategies : three strategies were identified ; education and justification campaign , nutritional intervention ( providing nutritious , safe and diverse snacks ) and networking . \n performing the interventions : interventions that were implementing in selected schools were providing a diverse and nutritious snack package along with nutrition education for both groups while the first group ( poor and malnourished students ) was utilized the package free of charge . \n education and justification intervention : regarding the literature review and expert opinion , an educational group affiliated with the advocacy team has prepared educational booklets about nutritional information for each level ( degree ) . \n accordingly , education of these booklets has been integrated into regular education of students and they educated and justified for better nutrition life - style . obviously , student 's families had remarkable effect on children 's food habit . \n it leads the educational group to hold several meeting with the student 's parents to justify them about the project and its benefit for their children . \n after these meetings , parental desire for participation in the project illustrated the effectiveness of the justification meeting with them . \n educate fifteen talk show programs in tv and radio , 12 published papers in the local newspaper , have implemented to mobilize the community and gain their support . \n healthy diet , the importance of breakfast and snack in adolescence , wrong food habits among school age children , role of the family to improve food habit of children were the main topics , in which media campaign has focused on . nutritional intervention : the snack basket of the students \n was replaced with traditional , nutritious and diverse foods . in general , the new snack package in average has provided 380 kcal energy , 15 g protein along with sufficient calcium and iron . \n low economic and malnourished children were supported by executive group affiliated with advocacy team and the rest of them prepare their snack by themselves . \n research and evaluation : in this step , the literacy and anthropometric indices ( bmi ) of students were assessed before and after the interventions . \n the reference for anthropometric measures was the world health organization / national center for health statistics ( who / nchs ) standards and the cut - offs were - two standard deviations ( sd ) from the mean . \n each student that was malnourished and poor has been taken into account for free food and nutritious snacks . \n demographic information , height , weight and knowledge of the students were measured by use of a validated and reliable ( cronbach 's alpha was 0.61 ) questionnaire . \n this project is granted by shiraz university of medical sciences , charities and welfare organization and education organization of fars province . \n statistical analyses were performed using the statistical package for the social sciences ( spss ) software , version 17.0 ( spss inc . , chicago , il , usa ) . \n the results are expressed as mean  sd and proportions as appropriated . in order to determine the effective variables on the malnutrition status \n paired t test was used to compare the end values with baseline ones in each group . \n two - sided p < 0.05 was considered to be statistically significant . in this project , \n the who z - score cut - offs used were as follow : using bmi - for - age z - scores ; overweight : > + 1 sd , i.e. , z - score > 1 ( equivalent to bmi 25 kg / m ) , obesity : > + 2 sd ( equivalent to bmi 30 \n kg / m ) , thinness : < 2 sd and severe thinness : < 3 sd . \n study population contains 2897 children ; 70.8% were primary school students and 29.2% were secondary school students . \n 2336 ( 80.5% ) out of total students were well - off and 561 children ( 19.5% ) were indigent . \n 19.5% of subjects were in case group ( n = 561 ) and 80.5% were in the control group ( n = 2336 ) . \n the mean of age in welfare group was 10.0  2.3 and 10.5  2.5 in non - welfare group . \n demographic characteristics of school aged children in shiraz , iran table 2 shows the frequency of subjects in different categories of bmi for age in non - welfare and welfare groups of school aged children separately among boys and girls before and after a nutrition intervention based on advocacy process model in shiraz , iran . \n the frequency of subjects with bmi lower than < 2 sd decreased significantly after intervention among non - welfare girls ( p < 0.01 ) . \n however , there were no significant decreases in the frequency of subjects with bmi lower than < 2 sd boys . \n when we assess the effect of intervention in total population without separating by sex groups , we found no significant change in this population [ table 3 ] . \n bmi for age for iranian students aged 7 - 14 years based on gender according to who growth standards 2007 bmi for age for iranian students aged 7 - 14 years according to who growth standards 2007 in non - welfare and welfare groups of total population table 4 has shown the prevalence of normal bmi , mild , moderate and severe malnutrition in non - welfare and welfare groups of school aged children separately among boys and girls before and after a nutrition intervention based on advocacy process model . according to this table \n there were no significant differences in the prevalence of mild , moderate and severe malnutrition among girls and boys . \n table 4 also shows the mean of all anthropometric indices changed significantly after intervention both among girls and boys . \n the pre- and post - test education assessment in both groups showed that the student 's average knowledge score has been significantly increased from 12.5  3.2 to 16.8  4.3 ( p < 0.0001 ) . \n bmi , height and weight in non - welfare and welfare groups of school aged children separately in males and females before and after a nutrition intervention based on advocacy process model in shiraz , iran according to study 's finding the odds ratio ( or ) of sever thinness and thinness in non - welfare compared with welfare is 3.5 ( or = 3.5 , confidence interval [ ci ] = 2.5 - 3.9 , p < 0.001 ) . \n furthermore , the finding showed or of overweight and obesity in welfare compared to non - welfare is 19.3 ( or = 19.3 , ci = 2.5 - 3.9 , p = 0.04 ) . \n the result of this community intervention study revealed that nutrition intervention based on advocacy program had been successful to reduce the prevalence of underweight among poor girls . \n this study shows determinant factor of nutritional status of school age children was their socio - economic level . according to our knowledge , \n this is the first study , which determines the effect of a community intervention based on advocacy process on the malnutrition indices in a big city ( shiraz ) in iran . \n the other program in iran ( nffp ) is specified to deprived area and is not conducted in big cities . \n allocating millions of dollars to nffp by government , selecting the malnourished students through an active screening system at primary and middle schools , paying attention of policy makers to student 's nutrition have provided the opportunity to combat the problem . however , negligence of under - poverty line , providing poor snacks in terms of nutritional value and lack of variety are the main defects of this program . \n advocacy by definition is a blending of science , ethics and politics for comprehensive approaching health issues . by using advocacy program in california among the high school students for improving their nutrition and physical activity \n angeles unified school district participants emphasized on nutrition classes for families as well as students in addition to other interventions . in the present study \n another study revealed that evaluability assessment gave stakeholders the opportunity to reflect on the project and its implementation issues . \n it seems that in iran , free food program among the students not only is needed in deprived areas , but also it should be performed in big cities such as shiraz . at baseline , \n no significant difference was founded among wealthy students between the pre- and post - nutritional status intervention . \n in contrast , the numbers of students who have malnutrition decreased from 44% to 39.4% , which was identified as a significant among impecunious girls students . \n there was also a significant increase in the proportion of children with bmi that was normal for age ( 2 to + 1 sd ) most of the published community interventions showed better results among females compared with males . \n this difference in the impact of nutritional interventions between male and female might be related to the different age of puberty in the female population compared to the male population . in the age range of the present study female \n although , there is no nffp in big cities of iran , there are some programs for improving the nutritional status such as providing free milk in schools . \n a recent publication has shown that school feeding programs focus on milk supplementation had beneficial effects on the physical function and school performances specifically among girls in iran . \n the results of the mentioned study showed an improvement in the weight of children , psychological test 's scores and the grade - point average following this school feeding program . \n the intervention in the present study had focused on the snack intake in the school time . \n there are some reports regarding the nutrition transition in iran , which shows the importance of nutrition intervention to provide more healthy eating dietary habits among welfare groups of adolescents . \n hence , nutrition intervention especially in the form of nutrition education is needed in big cities and among welfare children and adolescents . although a study among iranian adolescents showed that dietary behavior of adolescents does not accord to their knowledge , which emphasize on the necessity of community intervention programs . a recent study regarding the major dietary pattern among iranian children showed the presence of four major dietary patterns , in which fast food pattern and sweet pattern as two major dietary patterns can be mentioned among iranian children . in advocacy program audience 's analysis \n accordingly , one of the prominent strategies in this study was working with media and was meeting with parent - teacher association that both of them were secondary target audiences \n . we also took into account policy makers in different levels , from national to local as primary audiences . \n advocacy team had several meetings with management and planning organization at national level and education organization of the fars province as well as principal of the targeted schools . \n providing nutritious snacks need contribution of private sector such as food industries or factories , but their benefits should be warranted . \n another choice was community involvement ; which can be achieved by female health volunteers who are working with the health system . \n advocacy team by using the support of charities and female health volunteers could establish a local factory that produced student 's snacks based on the new definition . however , there are some challenges on the way of expanding this program . \n mass production of the proposed snacks according to different desires and cultures and getting involvement of food industries with respect to marketing issues is one of those challenges . \n moreover , providing a supportive environment in order to change the food habits of the students and their parents among the wide range of the population require a sustainable and continuous inter - sector collaboration . \n although in a limited number of schools , in our study , interventions and advocacy program was successful , expanding this model to another areas around the country depends on convincing the policy makers at national level . in this \n regard , advocacy team should prepare evidenced based profile and transitional planning to convince the policy makers for improving the rule and regulation of nffp . \n the same as this study in other studies have also emphasized that there must be efforts to strengthen the capacity within the schools to deal with the nutritional problems either overweight , obesity or malnutrition by using of educational and nutritional intervention . \n assessing the dietary adherence is very important in nutrition intervention among population . as this population was children and adolescents we had a limitation in the blood sample collection to assess the subject 's dietary adherence . \n furthermore , this intervention was only focused on the intake of snack in school time and we did not have comprehensive information on the dietary intake of children and adolescents after school all over the day . \n the investigators propose further investigation in different areas of the country based on socio - cultural differences in order to make necessary modification and adapt this model to other areas . \n regarding the nutritional needs of the school age children , provision of a good platform for implementing and expanding this efficient model to the whole country based upon the socio - economic situation of each region is advisable to the moh and the moe . \n community nutrition intervention based on the advocacy process model is effective on reducing the prevalence of underweight specifically among female school aged children.
    """
    triples = extractor.extract(article3)
    print("\nExtracted triples:")
    print(triples)
