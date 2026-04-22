import { ReverseFunction } from "./functions/reverse/reverse.js";

export class Module {
    
    /**
     * 
     * @param {ReverseFunction[]} composites 
     */
    static sequential(...composites) {

        for (let i = 0; i < composites.length-1; i++) {
            composites[i].composite =  composites[i+1];
        }

        return composites[0];
    }
}